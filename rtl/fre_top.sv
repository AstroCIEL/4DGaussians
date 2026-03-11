// FRE Top Module
// Foveated Rasterizing Engine with 16 parallel blending cores

`include "gs_types.sv"
import gs_types::*;

module fre_top (
    input  logic clk,
    input  logic rst_n,
    
    // Task inputs from WBS (16 cores)
    input  wbs_task_t                       task_i [0:NUM_CORES-1],
    input  logic [NUM_CORES-1:0]            task_valid_i,
    output logic [NUM_CORES-1:0]            task_ready_o,
    
    // Sorted chunks from HSE (16 cores)
    input  sorted_chunk_t                   sorted_chunk_i [0:NUM_CORES-1],
    input  logic [NUM_CORES-1:0]            sorted_chunk_valid_i,
    output logic [NUM_CORES-1:0]            sorted_chunk_ready_o,
    
    // Gaussian data inputs (from external memory)
    input  deformed_gaussian_t              gaussian_data_i [0:NUM_CORES-1],
    input  logic [NUM_CORES-1:0]            gaussian_valid_i,
    output logic [NUM_CORES-1:0]            gaussian_ready_o,
    
    // Output to frame buffer
    output pixel_data_t                     pixel_data_o,
    output logic [15:0]                     pixel_x_o,
    output logic [15:0]                     pixel_y_o,
    output logic                            pixel_valid_o,
    input  logic                            pixel_ready_i,
    
    // Core status to WBS
    output logic [NUM_CORES-1:0]            core_ready_o
);

    // Internal signals from blending cores to interpolation unit
    pixel_data_t bc_to_interp [0:NUM_CORES-1][0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [NUM_CORES-1:0] bc_to_interp_valid;
    logic [NUM_CORES-1:0] bc_to_interp_ready;
    downsample_rate_e downsample_rates [0:NUM_CORES-1];
    logic [TILE_ID_WIDTH-1:0] tile_ids [0:NUM_CORES-1];
    
    // Instantiate 16 blending cores
    genvar i;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : gen_blending_cores
            fre_bc u_blending_core (
                .clk(clk),
                .rst_n(rst_n),
                .task_i(task_i[i]),
                .task_valid_i(task_valid_i[i]),
                .task_ready_o(task_ready_o[i]),
                .sorted_chunk_i(sorted_chunk_i[i]),
                .sorted_chunk_valid_i(sorted_chunk_valid_i[i]),
                .sorted_chunk_ready_o(sorted_chunk_ready_o[i]),
                .gaussian_data_i(gaussian_data_i[i]),
                .gaussian_valid_i(gaussian_valid_i[i]),
                .gaussian_ready_o(gaussian_ready_o[i]),
                .pixel_data_o(bc_to_interp[i]),
                .pixel_valid_o(bc_to_interp_valid[i]),
                .pixel_ready_i(bc_to_interp_ready[i]),
                .core_ready_o(core_ready_o[i])
            );
            
            // Extract downsampling rate and tile ID from task
            always_ff @(posedge clk) begin
                if (task_valid_i[i] && task_ready_o[i]) begin
                    downsample_rates[i] <= task_i[i].downsample_rate;
                    tile_ids[i] <= task_i[i].tile_id;
                end
            end
        end
    endgenerate
    
    // Instantiate interpolation unit
    fre_interp u_interp (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_data_i(bc_to_interp),
        .pixel_valid_i(bc_to_interp_valid),
        .pixel_ready_o(bc_to_interp_ready),
        .downsample_rate_i(downsample_rates),
        .tile_id_i(tile_ids),
        .pixel_data_o(pixel_data_o),
        .pixel_x_o(pixel_x_o),
        .pixel_y_o(pixel_y_o),
        .pixel_valid_o(pixel_valid_o),
        .pixel_ready_i(pixel_ready_i)
    );
    
    // Clock gating for power optimization
    logic [NUM_CORES-1:0] core_clk_en;
    generate
        for (i = 0; i < NUM_CORES; i++) begin : gen_clk_gating
            assign core_clk_en[i] = task_valid_i[i] || sorted_chunk_valid_i[i] ||
                                     gaussian_valid_i[i] || bc_to_interp_valid[i];
            // Note: Actual clock gating would be implemented using library cells
        end
    endgenerate

endmodule
