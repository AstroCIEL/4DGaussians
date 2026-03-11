// WBS Top Module
// Workload Balancing Scheduler

`include "gs_types.sv"
import gs_types::*;

module wbs_top (
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
    
    // Core status from HSE
    input  logic [NUM_CORES-1:0]            hse_core_ready_i,
    
    // Core status from FRE
    input  logic [NUM_CORES-1:0]            fre_core_ready_i,
    
    // Task dispatch to HSE cores
    output wbs_task_t                       hse_task_o [0:NUM_CORES-1],
    output logic [NUM_CORES-1:0]            hse_task_valid_o,
    input  logic [NUM_CORES-1:0]            hse_task_ready_i,
    
    // Task dispatch to FRE cores
    output wbs_task_t                       fre_task_o [0:NUM_CORES-1],
    output logic [NUM_CORES-1:0]            fre_task_valid_o,
    input  logic [NUM_CORES-1:0]            fre_task_ready_i
);

    // Internal signals
    wbs_task_t load_gen_to_iq;
    logic load_gen_to_iq_valid;
    logic load_gen_to_iq_ready;
    
    wbs_task_t iq_to_arbiter [0:IQ_WINDOW_SIZE-1];
    logic iq_to_arbiter_valid;
    logic [4:0] iq_to_arbiter_count;
    
    wbs_task_t arbiter_to_dispatcher;
    logic arbiter_to_dispatcher_valid;
    logic [4:0] arbiter_selected_index;
    logic arbiter_read_ack;
    
    // Instantiate load generator
    wbs_load_gen u_load_gen (
        .clk(clk),
        .rst_n(rst_n),
        .tile_workload_i(tile_workload_i),
        .tile_workload_valid_i(tile_workload_valid_i),
        .tile_workload_ready_o(tile_workload_ready_o),
        .gaze_params_i(gaze_params_i),
        .screen_width_i(screen_width_i),
        .screen_height_i(screen_height_i),
        .task_o(load_gen_to_iq),
        .task_valid_o(load_gen_to_iq_valid),
        .task_ready_i(load_gen_to_iq_ready)
    );
    
    // Instantiate instruction queue
    wbs_iq u_iq (
        .clk(clk),
        .rst_n(rst_n),
        .task_i(load_gen_to_iq),
        .task_valid_i(load_gen_to_iq_valid),
        .task_ready_o(load_gen_to_iq_ready),
        .task_window_o(iq_to_arbiter),
        .window_valid_o(iq_to_arbiter_valid),
        .window_count_o(iq_to_arbiter_count),
        .arbiter_read_i(arbiter_read_ack)
    );
    
    // Instantiate arbiter
    wbs_arbiter u_arbiter (
        .clk(clk),
        .rst_n(rst_n),
        .task_window_i(iq_to_arbiter),
        .window_valid_i(iq_to_arbiter_valid),
        .window_count_i(iq_to_arbiter_count),
        .selected_task_o(arbiter_to_dispatcher),
        .selected_valid_o(arbiter_to_dispatcher_valid),
        .selected_index_o(arbiter_selected_index),
        .read_ack_i(arbiter_read_ack)
    );
    
    // Instantiate dispatcher
    wbs_dispatcher u_dispatcher (
        .clk(clk),
        .rst_n(rst_n),
        .task_i(arbiter_to_dispatcher),
        .task_valid_i(arbiter_to_dispatcher_valid),
        .task_ready_o(arbiter_read_ack),
        .hse_core_ready_i(hse_core_ready_i),
        .fre_core_ready_i(fre_core_ready_i),
        .hse_task_o(hse_task_o),
        .hse_task_valid_o(hse_task_valid_o),
        .hse_task_ready_i(hse_task_ready_i),
        .fre_task_o(fre_task_o),
        .fre_task_valid_o(fre_task_valid_o),
        .fre_task_ready_i(fre_task_ready_i)
    );
    
    // Clock gating for power optimization
    logic wbs_clk_en;
    assign wbs_clk_en = tile_workload_valid_i || arbiter_to_dispatcher_valid ||
                        |hse_task_valid_o || |fre_task_valid_o;
    
    // Note: Actual clock gating would be implemented using library cells

endmodule
