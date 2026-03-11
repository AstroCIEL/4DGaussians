// WBS Instruction Queue
// FIFO buffer storing active candidate window of size K=32

`include "gs_types.sv"
import gs_types::*;

module wbs_iq (
    input  logic clk,
    input  logic rst_n,
    
    // Input from load generator
    input  wbs_task_t                       task_i,
    input  logic                            task_valid_i,
    output logic                            task_ready_o,
    
    // Output to arbiter
    output wbs_task_t                       task_window_o [0:IQ_WINDOW_SIZE-1],
    output logic                            window_valid_o,
    output logic [4:0]                      window_count_o,
    
    // Arbiter read request
    input  logic                            arbiter_read_i
);

    // Instruction Queue: circular buffer
    wbs_task_t iq_buffer [0:IQ_WINDOW_SIZE-1];
    logic [4:0] write_ptr;
    logic [4:0] read_ptr;
    logic [4:0] count;
    logic full, empty;
    
    assign full = (count == IQ_WINDOW_SIZE);
    assign empty = (count == 0);
    
    // Write pointer update
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_ptr <= '0;
            read_ptr <= '0;
            count <= '0;
            for (int i = 0; i < IQ_WINDOW_SIZE; i++) begin
                iq_buffer[i] <= '0;
            end
        end else begin
            // Write operation
            if (task_valid_i && task_ready_o && !full) begin
                iq_buffer[write_ptr] <= task_i;
                write_ptr <= (write_ptr + 1) % IQ_WINDOW_SIZE;
                count <= count + 1;
            end
            
            // Read operation (arbiter consumes one task)
            if (arbiter_read_i && !empty) begin
                read_ptr <= (read_ptr + 1) % IQ_WINDOW_SIZE;
                count <= count - 1;
            end
        end
    end
    
    // Output window: provide all valid entries to arbiter
    always_comb begin
        window_count_o = count;
        window_valid_o = !empty;
        
        for (int i = 0; i < IQ_WINDOW_SIZE; i++) begin
            if (i < count) begin
                int idx = (read_ptr + i) % IQ_WINDOW_SIZE;
                task_window_o[i] = iq_buffer[idx];
            end else begin
                task_window_o[i] = '0;
            end
        end
    end
    
    assign task_ready_o = !full;

endmodule
