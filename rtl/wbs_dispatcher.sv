// WBS Task Dispatcher
// Monitors HSE/FRE core status and dispatches tasks

`include "gs_types.sv"
import gs_types::*;

module wbs_dispatcher (
    input  logic clk,
    input  logic rst_n,
    
    // Input from arbiter
    input  wbs_task_t                       task_i,
    input  logic                            task_valid_i,
    output logic                            task_ready_o,
    
    // Core status from HSE (16 cores)
    input  logic [NUM_CORES-1:0]            hse_core_ready_i,
    
    // Core status from FRE (16 cores)
    input  logic [NUM_CORES-1:0]            fre_core_ready_i,
    
    // Task dispatch to HSE cores
    output wbs_task_t                       hse_task_o [0:NUM_CORES-1],
    output logic [NUM_CORES-1:0]            hse_task_valid_o,
    input  logic [NUM_CORES-1:0]            hse_task_ready_i,
    
    // Task dispatch to FRE cores (same as HSE for paired cores)
    output wbs_task_t                       fre_task_o [0:NUM_CORES-1],
    output logic [NUM_CORES-1:0]            fre_task_valid_o,
    input  logic [NUM_CORES-1:0]            fre_task_ready_i
);

    // Core availability: both HSE and FRE must be ready for a core pair
    logic [NUM_CORES-1:0] core_available;
    logic [3:0] available_core_id;
    logic has_available_core;
    
    assign core_available = hse_core_ready_i & fre_core_ready_i;
    
    // Find first available core (priority encoder)
    always_comb begin
        has_available_core = 1'b0;
        available_core_id = '0;
        for (int i = 0; i < NUM_CORES; i++) begin
            if (core_available[i] && !has_available_core) begin
                has_available_core = 1'b1;
                available_core_id = i;
            end
        end
    end
    
    // Dispatch state machine
    typedef enum logic [1:0] {
        IDLE,
        DISPATCH,
        WAIT_ACK
    } dispatch_state_e;
    
    dispatch_state_e state, next_state;
    logic [3:0] dispatched_core;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            dispatched_core <= '0;
        end else begin
            state <= next_state;
            if (state == DISPATCH) begin
                dispatched_core <= available_core_id;
            end
        end
    end
    
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (task_valid_i && has_available_core) begin
                    next_state = DISPATCH;
                end
            end
            DISPATCH: begin
                next_state = WAIT_ACK;
            end
            WAIT_ACK: begin
                if (hse_task_ready_i[dispatched_core] && fre_task_ready_i[dispatched_core]) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Task dispatch outputs
    always_comb begin
        task_ready_o = (state == IDLE) && has_available_core;
        
        // Default: no task valid
        for (int i = 0; i < NUM_CORES; i++) begin
            hse_task_o[i] = '0;
            hse_task_valid_o[i] = 1'b0;
            fre_task_o[i] = '0;
            fre_task_valid_o[i] = 1'b0;
        end
        
        // Dispatch to selected core
        if (state == DISPATCH || state == WAIT_ACK) begin
            hse_task_o[dispatched_core] = task_i;
            hse_task_valid_o[dispatched_core] = 1'b1;
            fre_task_o[dispatched_core] = task_i;
            fre_task_valid_o[dispatched_core] = 1'b1;
        end
    end

endmodule
