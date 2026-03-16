// HSE Top Module
// Hierarchical Sorting Engine with 16 parallel cores

`include "gs_types.sv"
import gs_types::*;

module hse_top (
    input  logic clk,
    input  logic rst_n,
    
    // Task inputs from WBS (16 cores)
    input  wbs_task_t                       task_i [0:NUM_CORES-1],
    input  logic [NUM_CORES-1:0]            task_valid_i,
    output logic [NUM_CORES-1:0]            task_ready_o,
    
    // Gaussian data inputs (from external memory via AXI)
    input  deformed_gaussian_t              gaussian_data_i [0:NUM_CORES-1],
    input  logic [NUM_CORES-1:0]            gaussian_valid_i,
    output logic [NUM_CORES-1:0]            gaussian_ready_o,
    
    // Outputs: sorted chunks to FRE (16 cores)
    output sorted_chunk_t                   sorted_chunk_o [0:NUM_CORES-1],
    output logic [NUM_CORES-1:0]            sorted_chunk_valid_o,
    input  logic [NUM_CORES-1:0]            sorted_chunk_ready_i,
    
    // Core status to WBS
    output logic [NUM_CORES-1:0]            core_ready_o
);

    // Instantiate 16 HSE cores
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : gen_hse_cores
            hse_core u_hse_core (
                .clk(clk),
                .rst_n(rst_n),
                .task_i(task_i[i]),
                .task_valid_i(task_valid_i[i]),
                .task_ready_o(task_ready_o[i]),
                .gaussian_data_i(gaussian_data_i[i]),
                .gaussian_valid_i(gaussian_valid_i[i]),
                .gaussian_ready_o(gaussian_ready_o[i]),
                .sorted_chunk_o(sorted_chunk_o[i]),
                .sorted_chunk_valid_o(sorted_chunk_valid_o[i]),
                .sorted_chunk_ready_i(sorted_chunk_ready_i[i]),
                .core_ready_o(core_ready_o[i])
            );
        end
    endgenerate
    
    // Clock gating for power optimization
    logic [NUM_CORES-1:0] core_clk_en;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : gen_clk_gating
            assign core_clk_en[i] = task_valid_i[i] || sorted_chunk_valid_o[i] ||
                                     gaussian_valid_i[i];
            // Note: Actual clock gating would be implemented using library cells
        end
    endgenerate

endmodule
