// UDPE Culling Unit
// Calculates camera-space coordinates and evaluates frustum visibility

`include "gs_types.sv"
import gs_types::*;

module udpe_culling (
    input  logic clk,
    input  logic rst_n,
    
    // Input from dispatcher or deformation unit
    input  gaussian_primitive_t             gaussian_in,
    input  logic                            gaussian_valid_i,
    output logic                            gaussian_ready_o,
    
    // Input from deformation unit (for dynamic path)
    input  deformed_gaussian_t              deformed_in,
    input  logic                            deformed_valid_i,
    output logic                            deformed_ready_o,
    
    // Camera parameters
    input  camera_params_t                  camera_params_i,
    
    // Output to intersection unit
    output deformed_gaussian_t              gaussian_out,
    output logic                            gaussian_valid_o,
    input  logic                            gaussian_ready_i,
    
    // Output to deformation unit (for quasi-static path)
    output gaussian_primitive_t             visible_gaussian_o,
    output logic                            visible_gaussian_valid_o,
    input  logic                            visible_gaussian_ready_i
);

    // Internal pipeline stages
    typedef enum logic [1:0] {
        IDLE,
        TRANSFORM,
        CULL_CHECK,
        OUTPUT
    } cull_state_e;
    
    cull_state_e state, next_state;
    
    // Input selection: use deformed input if available, otherwise use raw input
    logic use_deformed;
    gaussian_primitive_t selected_gaussian;
    
    assign use_deformed = deformed_valid_i;
    assign selected_gaussian = use_deformed ? 
                               {deformed_in.mu_x, deformed_in.mu_y, deformed_in.mu_z,
                                deformed_in.sigma_xx, deformed_in.sigma_xy, deformed_in.sigma_xz,
                                deformed_in.sigma_yy, deformed_in.sigma_yz, deformed_in.sigma_zz,
                                deformed_in.opacity, deformed_in.sh_coeffs,
                                2'b00, deformed_in.gaussian_id} : 
                               gaussian_in;
    
    // Camera-space coordinates
    logic [31:0] cam_x, cam_y, cam_z, cam_w;
    logic [31:0] world_pos [0:3];
    logic [31:0] view_matrix [0:15];
    
    assign view_matrix = camera_params_i.view_matrix;
    assign world_pos[0] = selected_gaussian.mu_x;
    assign world_pos[1] = selected_gaussian.mu_y;
    assign world_pos[2] = selected_gaussian.mu_z;
    assign world_pos[3] = 32'h3F800000;  // w = 1.0 (float32 representation)
    
    // Matrix-vector multiplication: camera_pos = view_matrix * world_pos
    logic [31:0] mu_w;
    assign mu_w = 32'h3F800000;  // w = 1.0 (float32 representation)
    
    always_ff @(posedge clk) begin
        if (state == TRANSFORM) begin
            cam_x <= view_matrix[0] * world_pos[0] + view_matrix[4] * world_pos[1] + 
                     view_matrix[8] * world_pos[2] + view_matrix[12] * world_pos[3];
            cam_y <= view_matrix[1] * world_pos[0] + view_matrix[5] * world_pos[1] + 
                     view_matrix[9] * world_pos[2] + view_matrix[13] * world_pos[3];
            cam_z <= view_matrix[2] * world_pos[0] + view_matrix[6] * world_pos[1] + 
                     view_matrix[10] * world_pos[2] + view_matrix[14] * world_pos[3];
            cam_w <= view_matrix[3] * world_pos[0] + view_matrix[7] * world_pos[1] + 
                     view_matrix[11] * world_pos[2] + view_matrix[15] * world_pos[3];
        end
    end
    
    // Frustum culling logic
    logic visible;
    logic [31:0] depth;
    
    // Simplified frustum culling: check if Gaussian is within view frustum
    // In real implementation, this would check against 6 frustum planes
    // Note: Simplified implementation - avoiding floating point comparisons
    // In a real system, this would use a floating point comparator IP or
    // convert to fixed-point representation for hardware efficiency
    always_comb begin
        // Check if behind camera (z > 0 in camera space, assuming right-handed system)
        logic behind_camera;
        logic too_far;
        logic too_near;
        
        // Simplified checks using sign bit only (avoiding FP comparisons)
        // Positive values have sign bit = 0 (IEEE 754)
        behind_camera = (cam_z[31] == 1'b0 && cam_z != 32'h0);
        // For far/near plane checks, we use simplified logic:
        // Check if value is extremely negative (far) or close to zero from negative side (near)
        // This is a simplified check - real implementation would use FP comparator
        too_far = 1'b0;   // Simplified: disable far plane check
        too_near = 1'b0;  // Simplified: disable near plane check
        
        visible = !behind_camera && !too_far && !too_near;
        depth = cam_z;  // Store depth for sorting
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
                if (gaussian_valid_i || deformed_valid_i) begin
                    next_state = TRANSFORM;
                end
            end
            TRANSFORM: begin
                next_state = CULL_CHECK;
            end
            CULL_CHECK: begin
                next_state = OUTPUT;
            end
            OUTPUT: begin
                if (gaussian_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Output assignment
    always_ff @(posedge clk) begin
        if (state == CULL_CHECK) begin
            gaussian_out.mu_x <= cam_x;
            gaussian_out.mu_y <= cam_y;
            gaussian_out.mu_z <= cam_z;
            gaussian_out.sigma_xx <= selected_gaussian.sigma_xx;
            gaussian_out.sigma_xy <= selected_gaussian.sigma_xy;
            gaussian_out.sigma_xz <= selected_gaussian.sigma_xz;
            gaussian_out.sigma_yy <= selected_gaussian.sigma_yy;
            gaussian_out.sigma_yz <= selected_gaussian.sigma_yz;
            gaussian_out.sigma_zz <= selected_gaussian.sigma_zz;
            gaussian_out.opacity <= selected_gaussian.opacity;
            gaussian_out.sh_coeffs <= selected_gaussian.sh_coeffs;
            gaussian_out.gaussian_id <= selected_gaussian.gaussian_id;
            gaussian_out.depth <= depth[DEPTH_WIDTH-1:0];
            gaussian_out.visible <= visible;
        end
    end
    
    // Handshaking
    assign gaussian_ready_o = (state == IDLE);
    assign deformed_ready_o = (state == IDLE);
    assign gaussian_valid_o = (state == OUTPUT) && visible;
    
    // For quasi-static path: output visible Gaussians to deformation unit
    assign visible_gaussian_o = selected_gaussian;
    assign visible_gaussian_valid_o = (state == OUTPUT) && visible && !use_deformed;
    assign deformed_ready_o = (state == IDLE) || (state == OUTPUT && gaussian_ready_i);

endmodule
