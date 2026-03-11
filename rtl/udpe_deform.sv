// UDPE Deformation Unit
// Executes lightweight neural network to update Gaussian parameters

`include "gs_types.sv"
import gs_types::*;

module udpe_deform (
    input  logic clk,
    input  logic rst_n,
    
    // Input from dispatcher
    input  gaussian_primitive_t             gaussian_in,
    input  logic                            gaussian_valid_i,
    output logic                            gaussian_ready_o,
    
    // Camera parameters
    input  camera_params_t                  camera_params_i,
    
    // Output to culling unit
    output deformed_gaussian_t              gaussian_out,
    output logic                            gaussian_valid_o,
    input  logic                            gaussian_ready_i,
    
    // Weight cache interface (simplified - would connect to external memory)
    output logic [31:0]                     weight_addr_o,
    output logic                            weight_req_o,
    input  logic [WEIGHT_WIDTH-1:0]        weight_data_i [0:7],  // 8 weights per cycle
    input  logic                            weight_valid_i
);

    // Internal state
    typedef enum logic [2:0] {
        IDLE,
        FETCH_WEIGHTS,
        COMPUTE_DELTA,
        APPLY_DELTA,
        OUTPUT_RESULT
    } deform_state_e;
    
    deform_state_e state, next_state;
    
    // Feature buffer for neural network input
    logic [FEATURE_WIDTH-1:0] feature_buffer [0:15];
    logic [WEIGHT_WIDTH-1:0] weight_buffer [0:63];  // Cache for weights
    logic [4:0] weight_count;
    
    // Systolic array parameters (simplified MAC units)
    parameter int MAC_UNITS = 8;
    logic [31:0] mac_results [0:MAC_UNITS-1];
    logic [31:0] delta_mu [0:2];      // Delta for mean position
    logic [31:0] delta_sigma [0:5];   // Delta for covariance matrix
    logic [7:0]  delta_opacity;
    logic [95:0] delta_sh;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            weight_count <= '0;
        end else begin
            state <= next_state;
            if (state == FETCH_WEIGHTS && weight_valid_i) begin
                weight_count <= weight_count + 1;
            end else if (state == IDLE) begin
                weight_count <= '0;
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (gaussian_valid_i) begin
                    next_state = FETCH_WEIGHTS;
                end
            end
            FETCH_WEIGHTS: begin
                if (weight_count >= 7) begin  // Assuming 64 weights total
                    next_state = COMPUTE_DELTA;
                end
            end
            COMPUTE_DELTA: begin
                next_state = APPLY_DELTA;
            end
            APPLY_DELTA: begin
                next_state = OUTPUT_RESULT;
            end
            OUTPUT_RESULT: begin
                if (gaussian_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Feature extraction: concatenate mu and time_step
    always_ff @(posedge clk) begin
        if (state == IDLE && gaussian_valid_i) begin
            feature_buffer[0] <= gaussian_in.mu_x;
            feature_buffer[1] <= gaussian_in.mu_y;
            feature_buffer[2] <= gaussian_in.mu_z;
            feature_buffer[3] <= camera_params_i.time_step;
            // Additional features can be added here
        end
    end
    
    // Weight cache address generation
    assign weight_addr_o = gaussian_in.gaussian_id * 64 + weight_count * 8;  // Simplified addressing
    assign weight_req_o = (state == FETCH_WEIGHTS);
    
    // Simplified MAC computation (systolic array simulation)
    always_ff @(posedge clk) begin
        if (state == COMPUTE_DELTA) begin
            // Simplified neural network forward pass
            // In real implementation, this would be a proper systolic array
            for (int i = 0; i < MAC_UNITS; i++) begin
                mac_results[i] <= feature_buffer[i % 4] * weight_buffer[i];
            end
            
            // Compute deltas (simplified - actual implementation would use proper activation functions)
            delta_mu[0] <= mac_results[0] + mac_results[1];
            delta_mu[1] <= mac_results[2] + mac_results[3];
            delta_mu[2] <= mac_results[4] + mac_results[5];
            
            delta_sigma[0] <= mac_results[6];
            delta_sigma[1] <= mac_results[7];
            // ... more delta computations
        end
    end
    
    // Apply deltas to canonical Gaussian parameters
    always_ff @(posedge clk) begin
        if (state == APPLY_DELTA) begin
            gaussian_out.mu_x <= gaussian_in.mu_x + delta_mu[0];
            gaussian_out.mu_y <= gaussian_in.mu_y + delta_mu[1];
            gaussian_out.mu_z <= gaussian_in.mu_z + delta_mu[2];
            
            gaussian_out.sigma_xx <= gaussian_in.sigma_xx + delta_sigma[0];
            gaussian_out.sigma_xy <= gaussian_in.sigma_xy + delta_sigma[1];
            // ... apply all deltas
            
            gaussian_out.opacity <= gaussian_in.opacity + delta_opacity;
            gaussian_out.sh_coeffs <= gaussian_in.sh_coeffs + delta_sh;
            gaussian_out.gaussian_id <= gaussian_in.gaussian_id;
            
            // Depth and visibility will be computed in culling unit
            gaussian_out.depth <= '0;
            gaussian_out.visible <= 1'b0;
        end
    end
    
    // Output handshaking
    assign gaussian_ready_o = (state == IDLE);
    assign gaussian_valid_o = (state == OUTPUT_RESULT);
    
    // Weight buffer update
    always_ff @(posedge clk) begin
        if (weight_valid_i && state == FETCH_WEIGHTS) begin
            for (int i = 0; i < 8; i++) begin
                weight_buffer[weight_count * 8 + i] <= weight_data_i[i];
            end
        end
    end

endmodule
