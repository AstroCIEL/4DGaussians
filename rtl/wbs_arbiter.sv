// WBS Tile Arbiter
// Combinational comparator tree to find tile with maximum workload

`include "gs_types.sv"
import gs_types::*;

module wbs_arbiter (
    input  logic clk,
    input  logic rst_n,
    
    // Input window from IQ
    input  wbs_task_t                       task_window_i [0:IQ_WINDOW_SIZE-1],
    input  logic                            window_valid_i,
    input  logic [4:0]                      window_count_i,
    
    // Output: selected task with maximum workload
    output wbs_task_t                       selected_task_o,
    output logic                            selected_valid_o,
    output logic [4:0]                      selected_index_o,
    
    // Acknowledge read
    input  logic                            read_ack_i
);

    // Combinational comparator tree
    // Compare all tasks in parallel and find maximum workload
    logic [15:0] workloads [0:IQ_WINDOW_SIZE-1];
    logic [IQ_WINDOW_SIZE-1:0] valid_mask;
    logic [4:0] max_index;
    logic [15:0] max_workload;
    
    // Extract workloads and create valid mask
    always_comb begin
        for (int i = 0; i < IQ_WINDOW_SIZE; i++) begin
            workloads[i] = task_window_i[i].workload;
            valid_mask[i] = (i < window_count_i);
        end
    end
    
    // Recursive comparator tree to find maximum
    // This is a simplified version - real implementation would use proper tree structure
    function automatic logic [4:0] find_max_index(
        input logic [15:0] workloads [0:IQ_WINDOW_SIZE-1],
        input logic [IQ_WINDOW_SIZE-1:0] valid_mask
    );
        logic [15:0] max_val;
        logic [4:0] max_idx;
        max_val = 0;
        max_idx = 0;
        
        for (int i = 0; i < IQ_WINDOW_SIZE; i++) begin
            if (valid_mask[i] && workloads[i] > max_val) begin
                max_val = workloads[i];
                max_idx = i;
            end
        end
        
        return max_idx;
    endfunction
    
    // Find maximum workload task
    always_comb begin
        max_index = find_max_index(workloads, valid_mask);
        max_workload = workloads[max_index];
    end
    
    // Output selected task
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            selected_task_o <= '0;
            selected_valid_o <= 1'b0;
            selected_index_o <= '0;
        end else begin
            if (window_valid_i && window_count_i > 0) begin
                selected_task_o <= task_window_i[max_index];
                selected_valid_o <= 1'b1;
                selected_index_o <= max_index;
            end else begin
                selected_valid_o <= 1'b0;
            end
            
            if (read_ack_i) begin
                selected_valid_o <= 1'b0;
            end
        end
    end

endmodule
