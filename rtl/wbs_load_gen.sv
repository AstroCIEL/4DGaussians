// WBS Load Generator
// Estimates workload and applies MAFR downsampling rates

`include "gs_types.sv"
import gs_types::*;

module wbs_load_gen (
    input  logic clk,
    input  logic rst_n,
    
    // Input from UDPE
    input  tile_workload_t                  tile_workload_i,
    input  logic                            tile_workload_valid_i,
    output logic                            tile_workload_ready_o,
    
    // Gaze parameters
    input  gaze_params_t                    gaze_params_i,
    
    // Screen resolution
    input  logic [15:0]                     screen_width_i,
    input  logic [15:0]                     screen_height_i,
    
    // Output tasks following Hilbert curve sequence
    output wbs_task_t                       task_o,
    output logic                            task_valid_o,
    input  logic                            task_ready_i
);

    // Tile workload accumulator (per tile)
    logic [15:0] tile_workloads [0:1023];  // Assuming max 1024 tiles
    logic [GAUSSIAN_ID_WIDTH-1:0] tile_gaussian_counts [0:1023];
    logic [GAUSSIAN_ID_WIDTH-1:0] tile_start_ids [0:1023];
    logic tile_valid [0:1023];
    
    // Hilbert curve sequence generator
    logic [TILE_ID_WIDTH-1:0] hilbert_sequence [0:IQ_WINDOW_SIZE-1];
    logic [4:0] hilbert_index;
    
    // Compute tile coordinates from tile_id
    logic [4:0] tile_x, tile_y;
    assign tile_x = tile_workload_i.tile_id % (screen_width_i / TILE_SIZE);
    assign tile_y = tile_workload_i.tile_id / (screen_width_i / TILE_SIZE);
    
    // Compute screen-space coordinates of tile center
    logic [15:0] tile_center_x, tile_center_y;
    assign tile_center_x = tile_x * TILE_SIZE + TILE_SIZE / 2;
    assign tile_center_y = tile_y * TILE_SIZE + TILE_SIZE / 2;
    
    // Compute eccentricity from gaze point
    logic [31:0] eccentricity;
    logic [31:0] dx, dy;
    assign dx = (tile_center_x > gaze_params_i.gaze_x) ? 
                (tile_center_x - gaze_params_i.gaze_x) : 
                (gaze_params_i.gaze_x - tile_center_x);
    assign dy = (tile_center_y > gaze_params_i.gaze_y) ? 
                (tile_center_y - gaze_params_i.gaze_y) : 
                (gaze_params_i.gaze_y - tile_center_y);
    assign eccentricity = dx * dx + dy * dy;  // Squared distance
    
    // Determine downsampling rate based on eccentricity
    downsample_rate_e downsample_rate;
    always_comb begin
        if (eccentricity < gaze_params_i.fovea_radius * gaze_params_i.fovea_radius) begin
            downsample_rate = DOWNSAMPLE_1X;
        end else if (eccentricity < (gaze_params_i.fovea_radius * 2) * (gaze_params_i.fovea_radius * 2)) begin
            downsample_rate = DOWNSAMPLE_2X;
        end else begin
            downsample_rate = DOWNSAMPLE_4X;
        end
    end
    
    // Accumulate workload per tile
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (integer i = 0; i < 1024; i++) begin
                tile_workloads[i] <= '0;
                tile_gaussian_counts[i] <= '0;
                tile_start_ids[i] <= '0;
                tile_valid[i] <= 1'b0;
            end
        end else begin
            if (tile_workload_valid_i && tile_workload_ready_o) begin
                tile_workloads[tile_workload_i.tile_id] <= 
                    tile_workloads[tile_workload_i.tile_id] + tile_workload_i.workload_estimate;
                
                if (!tile_valid[tile_workload_i.tile_id]) begin
                    tile_start_ids[tile_workload_i.tile_id] <= tile_workload_i.gaussian.gaussian_id;
                    tile_valid[tile_workload_i.tile_id] <= 1'b1;
                end
                
                tile_gaussian_counts[tile_workload_i.tile_id] <= 
                    tile_gaussian_counts[tile_workload_i.tile_id] + 1;
            end
        end
    end
    
    // Simplified Hilbert curve generation (2D space-filling curve)
    // In real implementation, this would be a proper Hilbert curve generator
    // Note: Using fixed iteration count to avoid synthesis loop unrolling issues
    // Maximum iterations: 10 (sufficient for screen_width up to 32768 pixels with TILE_SIZE=32)
    function automatic logic [TILE_ID_WIDTH-1:0] hilbert_xy_to_d(integer n, integer x, integer y);
        logic [TILE_ID_WIDTH-1:0] d;
        integer rx, ry, t, s;
        logic done;
        d = 0;
        s = n / 2;  // Initialize s before loop
        done = 1'b0;
        // Use fixed iteration count instead of while loop for synthesis
        for (integer iter = 0; iter < 10; iter++) begin
            if (!done && s > 0) begin
                rx = (x & s) > 0;
                ry = (y & s) > 0;
                d = d + s * s * ((3 * rx) ^ ry);
                if (ry == 0) begin
                    if (rx == 1) begin
                        x = n - 1 - x;
                        y = n - 1 - y;
                    end
                    t = x;
                    x = y;
                    y = t;
                end
                s = s / 2;  // Update s at end of loop
                if (s == 0) begin
                    done = 1'b1;  // Mark as done when s becomes 0
                end
            end
        end
        return d;
    endfunction
    
    // Generate Hilbert sequence (simplified - would be precomputed)
    integer tiles_per_dim;
    always_comb begin
        tiles_per_dim = screen_width_i / TILE_SIZE;
    end
    
    always_ff @(posedge clk) begin
        for (integer i = 0; i < IQ_WINDOW_SIZE; i++) begin
            automatic integer hilbert_x = i % tiles_per_dim;
            automatic integer hilbert_y = i / tiles_per_dim;
            hilbert_sequence[i] <= hilbert_xy_to_d(tiles_per_dim, hilbert_x, hilbert_y);
        end
    end
    
    // Output task generation
    typedef enum logic [1:0] {
        IDLE,
        GENERATE_TASK,
        OUTPUT_TASK
    } gen_state_e;
    
    gen_state_e state, next_state;
    logic [TILE_ID_WIDTH-1:0] current_tile_id;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            hilbert_index <= '0;
            current_tile_id <= '0;
        end else begin
            state <= next_state;
            if (state == GENERATE_TASK && task_ready_i) begin
                hilbert_index <= hilbert_index + 1;
                if (hilbert_index == IQ_WINDOW_SIZE - 1) begin
                    hilbert_index <= '0;
                end
            end
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (tile_valid[hilbert_sequence[hilbert_index]]) begin
                    next_state = GENERATE_TASK;
                end
            end
            GENERATE_TASK: begin
                next_state = OUTPUT_TASK;
            end
            OUTPUT_TASK: begin
                if (task_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Task output
    always_ff @(posedge clk) begin
        if (state == GENERATE_TASK) begin
            current_tile_id <= hilbert_sequence[hilbert_index];
            task_o.tile_id <= hilbert_sequence[hilbert_index];
            task_o.workload <= tile_workloads[hilbert_sequence[hilbert_index]];
            task_o.downsample_rate <= downsample_rate;
            task_o.start_gaussian_id <= tile_start_ids[hilbert_sequence[hilbert_index]];
            task_o.num_gaussians <= tile_gaussian_counts[hilbert_sequence[hilbert_index]];
        end
    end
    
    assign tile_workload_ready_o = 1'b1;  // Always ready to accept
    assign task_valid_o = (state == OUTPUT_TASK);

endmodule
