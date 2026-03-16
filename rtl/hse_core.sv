// HSE Core
// Two-stage sorting: Quick Sort + Bitonic Sort

`include "gs_types.sv"
import gs_types::*;

module hse_core (
    input  logic clk,
    input  logic rst_n,
    
    // Task input from WBS
    input  wbs_task_t                       task_i,
    input  logic                            task_valid_i,
    output logic                            task_ready_o,
    
    // Gaussian data input (from external memory via AXI)
    input  deformed_gaussian_t              gaussian_data_i,
    input  logic                            gaussian_valid_i,
    output logic                            gaussian_ready_o,
    
    // Output: sorted index chunks to FRE
    output sorted_chunk_t                   sorted_chunk_o,
    output logic                            sorted_chunk_valid_o,
    input  logic                            sorted_chunk_ready_i,
    
    // Core status
    output logic                            core_ready_o
);

    // Local memory for Gaussian IDs and depths
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_ids [0:1023];  // Max Gaussians per tile
    logic [DEPTH_WIDTH-1:0] depths [0:1023];
    logic [GAUSSIAN_ID_WIDTH-1:0] num_gaussians;
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_count;
    
    // Sorting state machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_DATA,
        QUICK_SORT,
        BITONIC_SORT,
        OUTPUT_CHUNKS
    } sort_state_e;
    
    sort_state_e state, next_state;
    
    // Quick sort pivot and partition
    logic [GAUSSIAN_ID_WIDTH-1:0] pivot_idx;
    logic [GAUSSIAN_ID_WIDTH-1:0] left_ptr, right_ptr;
    logic [DEPTH_WIDTH-1:0] pivot_depth;
    
    // Bitonic sort stage
    logic [GAUSSIAN_ID_WIDTH-1:0] bitonic_stage;
    logic [GAUSSIAN_ID_WIDTH-1:0] chunk_idx;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            gaussian_count <= '0;
            num_gaussians <= '0;
            chunk_idx <= '0;
        end else begin
            state <= next_state;
            
            if (state == LOAD_DATA && gaussian_valid_i && gaussian_ready_o) begin
                gaussian_ids[gaussian_count] <= gaussian_data_i.gaussian_id;
                depths[gaussian_count] <= gaussian_data_i.depth;
                gaussian_count <= gaussian_count + 1;
            end
            
            if (state == IDLE && task_valid_i) begin
                num_gaussians <= task_i.num_gaussians;
                gaussian_count <= '0;
            end
            
            if (state == OUTPUT_CHUNKS && sorted_chunk_ready_i && sorted_chunk_valid_o) begin
                chunk_idx <= chunk_idx + 1;
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (task_valid_i) begin
                    next_state = LOAD_DATA;
                end
            end
            LOAD_DATA: begin
                if (gaussian_count >= num_gaussians) begin
                    next_state = QUICK_SORT;
                end
            end
            QUICK_SORT: begin
                // Simplified: assume quick sort completes in fixed cycles
                // Real implementation would have proper partition logic
                next_state = BITONIC_SORT;
            end
            BITONIC_SORT: begin
                // Simplified: assume bitonic sort completes in fixed cycles
                // Real implementation would have proper bitonic network
                next_state = OUTPUT_CHUNKS;
            end
            OUTPUT_CHUNKS: begin
                if (chunk_idx * 16 >= num_gaussians && sorted_chunk_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Simplified Quick Sort (coarse sorting)
    // Real implementation would use proper recursive partitioning
    always_ff @(posedge clk) begin
        if (state == QUICK_SORT) begin
            // Simplified: just perform a basic partition
            // In real implementation, this would be a proper quicksort algorithm
            pivot_idx <= num_gaussians / 2;
            pivot_depth <= depths[pivot_idx];
        end
    end
    
    // Simplified Bitonic Sort (fine-grained sorting)
    // Real implementation would use proper bitonic merge network
    always_ff @(posedge clk) begin
        if (state == BITONIC_SORT) begin
            // Simplified: assume sorting happens here
            // Real implementation would have proper bitonic network stages
            bitonic_stage <= bitonic_stage + 1;
        end else begin
            bitonic_stage <= '0;
        end
    end
    
    // Output sorted chunks
    always_comb begin
        sorted_chunk_o.tile_id = task_i.tile_id;
        sorted_chunk_o.num_valid = (chunk_idx * 16 + 16 <= num_gaussians) ? 
                                   16 : (num_gaussians - chunk_idx * 16);
        
        for (int i = 0; i < 16; i++) begin
            automatic int idx = chunk_idx * 16 + i;
            if (idx < num_gaussians) begin
                sorted_chunk_o.gaussian_ids[i] = gaussian_ids[idx];
            end else begin
                sorted_chunk_o.gaussian_ids[i] = '0;
            end
        end
    end
    
    // Handshaking
    assign task_ready_o = (state == IDLE);
    assign gaussian_ready_o = (state == LOAD_DATA);
    assign sorted_chunk_valid_o = (state == OUTPUT_CHUNKS);
    assign core_ready_o = (state == IDLE);

endmodule
