// UDPE 2D Feature & Intersection Unit
// Computes screen-space features and tile intersections

`include "gs_types.sv"
import gs_types::*;

module udpe_intersect (
    input  logic clk,
    input  logic rst_n,
    
    // Input from culling unit
    input  deformed_gaussian_t              gaussian_in,
    input  logic                            gaussian_valid_i,
    output logic                            gaussian_ready_o,
    
    // Camera parameters
    input  camera_params_t                  camera_params_i,
    
    // Screen resolution parameters
    input  logic [15:0]                     screen_width_i,
    input  logic [15:0]                     screen_height_i,
    
    // Output to WBS
    output tile_workload_t                  tile_workload_o,
    output logic                            tile_workload_valid_o,
    input  logic                            tile_workload_ready_i
);

    // Projection parameters
    logic [31:0] proj_matrix [0:15];
    assign proj_matrix = camera_params_i.proj_matrix;
    
    // Clip-space coordinates
    logic [31:0] clip_x, clip_y, clip_z, clip_w;
    
    // Screen-space coordinates
    logic [31:0] screen_x, screen_y;
    logic [15:0] screen_x_int, screen_y_int;
    
    // Tile intersection
    logic [TILE_ID_WIDTH-1:0] tile_id;
    logic [4:0] tile_x, tile_y;  // Tile coordinates (assuming 32x32 tiles)
    
    // Workload estimate (based on Gaussian size in screen space)
    logic [15:0] workload_estimate;
    
    // Projection pipeline
    typedef enum logic [1:0] {
        IDLE,
        PROJECT,
        INTERSECT,
        OUTPUT
    } intersect_state_e;
    
    intersect_state_e state, next_state;
    
    // Camera-space w coordinate (always 1.0)
    logic [31:0] cam_w;
    assign cam_w = 32'h3F800000;  // w = 1.0 (float32 representation)
    
    // Projection: clip_pos = proj_matrix * camera_pos
    always_ff @(posedge clk) begin
        if (state == PROJECT) begin
            clip_x <= proj_matrix[0] * gaussian_in.mu_x + 
                      proj_matrix[4] * gaussian_in.mu_y + 
                      proj_matrix[8] * gaussian_in.mu_z + 
                      proj_matrix[12] * cam_w;
            clip_y <= proj_matrix[1] * gaussian_in.mu_x + 
                      proj_matrix[5] * gaussian_in.mu_y + 
                      proj_matrix[9] * gaussian_in.mu_z + 
                      proj_matrix[13] * cam_w;
            clip_z <= proj_matrix[2] * gaussian_in.mu_x + 
                      proj_matrix[6] * gaussian_in.mu_y + 
                      proj_matrix[10] * gaussian_in.mu_z + 
                      proj_matrix[14] * cam_w;
            clip_w <= proj_matrix[3] * gaussian_in.mu_x + 
                      proj_matrix[7] * gaussian_in.mu_y + 
                      proj_matrix[11] * gaussian_in.mu_z + 
                      proj_matrix[15] * cam_w;
        end
    end
    
    // Perspective divide and viewport transform
    always_ff @(posedge clk) begin
        if (state == INTERSECT && clip_w != 0) begin
            // Perspective divide
            screen_x <= (clip_x / clip_w + 1.0) * 0.5 * screen_width_i;
            screen_y <= (1.0 - clip_y / clip_w) * 0.5 * screen_height_i;
            
            screen_x_int <= screen_x[15:0];
            screen_y_int <= screen_y[15:0];
        end
    end
    
    // Tile intersection calculation
    always_comb begin
        tile_x = screen_x_int / TILE_SIZE;
        tile_y = screen_y_int / TILE_SIZE;
        tile_id = tile_y * (screen_width_i / TILE_SIZE) + tile_x;
    end
    
    // Workload estimate: based on Gaussian size in screen space
    // Simplified: use covariance matrix determinant as proxy for size
    always_comb begin
        // Estimate workload based on Gaussian screen-space extent
        // In real implementation, this would compute actual screen-space size
        logic [31:0] sigma_det;
        sigma_det = gaussian_in.sigma_xx * gaussian_in.sigma_yy * gaussian_in.sigma_zz;
        workload_estimate = sigma_det[15:0];  // Simplified workload metric
    end
    
    // State machine
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (gaussian_valid_i && gaussian_in.visible) begin
                    next_state = PROJECT;
                end
            end
            PROJECT: begin
                next_state = INTERSECT;
            end
            INTERSECT: begin
                next_state = OUTPUT;
            end
            OUTPUT: begin
                if (tile_workload_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Output assignment
    always_ff @(posedge clk) begin
        if (state == INTERSECT) begin
            tile_workload_o.gaussian <= gaussian_in;
            tile_workload_o.tile_id <= tile_id;
            tile_workload_o.workload_estimate <= workload_estimate;
        end
    end
    
    // Handshaking
    assign gaussian_ready_o = (state == IDLE);
    assign tile_workload_valid_o = (state == OUTPUT);

endmodule
