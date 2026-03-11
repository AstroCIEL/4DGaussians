// UDPE Top Module
// Unified Deformation-Preprocess Engine

`include "gs_types.sv"
import gs_types::*;

module udpe_top (
    input  logic clk,
    input  logic rst_n,
    
    // Input FIFO interface
    input  logic [GAUSSIAN_DATA_WIDTH-1:0] input_fifo_data_i,
    input  logic                            input_fifo_valid_i,
    output logic                            input_fifo_ready_o,
    
    // Camera parameters
    input  camera_params_t                  camera_params_i,
    
    // Screen resolution
    input  logic [15:0]                     screen_width_i,
    input  logic [15:0]                     screen_height_i,
    
    // Weight cache interface (to external memory)
    output logic [31:0]                     weight_addr_o,
    output logic                            weight_req_o,
    input  logic [WEIGHT_WIDTH-1:0]        weight_data_i [0:7],
    input  logic                            weight_valid_i,
    
    // Output to WBS
    output tile_workload_t                  tile_workload_o,
    output logic                            tile_workload_valid_o,
    input  logic                            tile_workload_ready_i
);

    // Internal signals between sub-modules
    gaussian_primitive_t dispatcher_to_deform;
    logic dispatcher_to_deform_valid;
    logic dispatcher_to_deform_ready;
    
    gaussian_primitive_t dispatcher_to_cull;
    logic dispatcher_to_cull_valid;
    logic dispatcher_to_cull_ready;
    
    deformed_gaussian_t deform_to_dispatcher;
    logic deform_to_dispatcher_valid;
    logic deform_to_dispatcher_ready;
    
    gaussian_primitive_t cull_to_dispatcher;
    logic cull_to_dispatcher_valid;
    logic cull_to_dispatcher_ready;
    
    deformed_gaussian_t cull_to_intersect;
    logic cull_to_intersect_valid;
    logic cull_to_intersect_ready;
    
    gaussian_primitive_t cull_to_deform;
    logic cull_to_deform_valid;
    logic cull_to_deform_ready;
    
    // Instantiate dispatcher
    udpe_dispatcher u_dispatcher (
        .clk(clk),
        .rst_n(rst_n),
        .input_fifo_data_i(input_fifo_data_i),
        .input_fifo_valid_i(input_fifo_valid_i),
        .input_fifo_ready_o(input_fifo_ready_o),
        .deform_data_o(dispatcher_to_deform),
        .deform_valid_o(dispatcher_to_deform_valid),
        .deform_ready_i(dispatcher_to_deform_ready),
        .cull_data_o(dispatcher_to_cull),
        .cull_valid_o(dispatcher_to_cull_valid),
        .cull_ready_i(dispatcher_to_cull_ready),
        .deform_to_cull_data_i(deform_to_dispatcher),
        .deform_to_cull_valid_i(deform_to_dispatcher_valid),
        .deform_to_cull_ready_o(deform_to_dispatcher_ready),
        .cull_to_deform_data_i(cull_to_dispatcher),
        .cull_to_deform_valid_i(cull_to_dispatcher_valid),
        .cull_to_deform_ready_o(cull_to_dispatcher_ready)
    );
    
    // Instantiate deformation unit
    udpe_deform u_deform (
        .clk(clk),
        .rst_n(rst_n),
        .gaussian_in(dispatcher_to_deform),
        .gaussian_valid_i(dispatcher_to_deform_valid),
        .gaussian_ready_o(dispatcher_to_deform_ready),
        .camera_params_i(camera_params_i),
        .gaussian_out(deform_to_dispatcher),
        .gaussian_valid_o(deform_to_dispatcher_valid),
        .gaussian_ready_i(deform_to_dispatcher_ready),
        .weight_addr_o(weight_addr_o),
        .weight_req_o(weight_req_o),
        .weight_data_i(weight_data_i),
        .weight_valid_i(weight_valid_i)
    );
    
    // Instantiate culling unit
    udpe_culling u_culling (
        .clk(clk),
        .rst_n(rst_n),
        .gaussian_in(dispatcher_to_cull),
        .gaussian_valid_i(dispatcher_to_cull_valid),
        .gaussian_ready_o(dispatcher_to_cull_ready),
        .deformed_in(deform_to_dispatcher),
        .deformed_valid_i(deform_to_dispatcher_valid),
        .deformed_ready_o(deform_to_dispatcher_ready),
        .camera_params_i(camera_params_i),
        .gaussian_out(cull_to_intersect),
        .gaussian_valid_o(cull_to_intersect_valid),
        .gaussian_ready_i(cull_to_intersect_ready),
        .visible_gaussian_o(cull_to_deform),
        .visible_gaussian_valid_o(cull_to_deform_valid),
        .visible_gaussian_ready_i(cull_to_deform_ready)
    );
    
    // Instantiate intersection unit
    udpe_intersect u_intersect (
        .clk(clk),
        .rst_n(rst_n),
        .gaussian_in(cull_to_intersect),
        .gaussian_valid_i(cull_to_intersect_valid),
        .gaussian_ready_o(cull_to_intersect_ready),
        .camera_params_i(camera_params_i),
        .screen_width_i(screen_width_i),
        .screen_height_i(screen_height_i),
        .tile_workload_o(tile_workload_o),
        .tile_workload_valid_o(tile_workload_valid_o),
        .tile_workload_ready_i(tile_workload_ready_i)
    );
    
    // Clock gating for power optimization
    logic udpe_clk_en;
    assign udpe_clk_en = input_fifo_valid_i || tile_workload_valid_o || 
                         dispatcher_to_deform_valid || dispatcher_to_cull_valid ||
                         cull_to_intersect_valid;
    
    // Note: Actual clock gating would be implemented using library cells
    // This is a placeholder for the concept

endmodule
