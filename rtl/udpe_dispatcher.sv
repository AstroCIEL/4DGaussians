// UDPE Parsing Dispatcher
// Routes Gaussian primitives based on motion tags

`include "gs_types.sv"
import gs_types::*;

module udpe_dispatcher (
    input  logic clk,
    input  logic rst_n,
    
    // Input FIFO interface
    input  logic [GAUSSIAN_DATA_WIDTH-1:0] input_fifo_data_i,
    input  logic                            input_fifo_valid_i,
    output logic                            input_fifo_ready_o,
    
    // To Deformation Unit
    output gaussian_primitive_t             deform_data_o,
    output logic                            deform_valid_o,
    input  logic                            deform_ready_i,
    
    // To Culling Unit (bypass deformation)
    output gaussian_primitive_t             cull_data_o,
    output logic                            cull_valid_o,
    input  logic                            cull_ready_i,
    
    // From Deformation Unit (to Culling)
    input  deformed_gaussian_t              deform_to_cull_data_i,
    input  logic                            deform_to_cull_valid_i,
    output logic                            deform_to_cull_ready_o,
    
    // From Culling Unit (to Deformation - quasi-static path)
    input  gaussian_primitive_t             cull_to_deform_data_i,
    input  logic                            cull_to_deform_valid_i,
    output logic                            cull_to_deform_ready_o
);

    // Extract motion tag from input data
    logic [1:0] motion_tag;
    gaussian_primitive_t gaussian_in;
    
    assign motion_tag = input_fifo_data_i[1:0];
    assign gaussian_in = input_fifo_data_i[GAUSSIAN_DATA_WIDTH-1:0];
    
    // State machine for routing logic
    typedef enum logic [1:0] {
        IDLE,
        ROUTE_STATIC,
        ROUTE_DYNAMIC,
        ROUTE_QUASI_STATIC
    } route_state_e;
    
    route_state_e state, next_state;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
        end else begin
            state <= next_state;
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (input_fifo_valid_i) begin
                    case (motion_tag)
                        MOTION_STATIC:       next_state = ROUTE_STATIC;
                        MOTION_DYNAMIC:      next_state = ROUTE_DYNAMIC;
                        MOTION_QUASI_STATIC: next_state = ROUTE_QUASI_STATIC;
                        default:             next_state = IDLE;
                    endcase
                end
            end
            ROUTE_STATIC: begin
                if (cull_ready_i && input_fifo_valid_i) begin
                    case (motion_tag)
                        MOTION_STATIC:       next_state = ROUTE_STATIC;
                        MOTION_DYNAMIC:      next_state = ROUTE_DYNAMIC;
                        MOTION_QUASI_STATIC: next_state = ROUTE_QUASI_STATIC;
                        default:             next_state = IDLE;
                    endcase
                end
            end
            ROUTE_DYNAMIC: begin
                if (deform_ready_i && input_fifo_valid_i) begin
                    case (motion_tag)
                        MOTION_STATIC:       next_state = ROUTE_STATIC;
                        MOTION_DYNAMIC:      next_state = ROUTE_DYNAMIC;
                        MOTION_QUASI_STATIC: next_state = ROUTE_QUASI_STATIC;
                        default:             next_state = IDLE;
                    endcase
                end
            end
            ROUTE_QUASI_STATIC: begin
                if (cull_ready_i && input_fifo_valid_i) begin
                    case (motion_tag)
                        MOTION_STATIC:       next_state = ROUTE_STATIC;
                        MOTION_DYNAMIC:      next_state = ROUTE_DYNAMIC;
                        MOTION_QUASI_STATIC: next_state = ROUTE_QUASI_STATIC;
                        default:             next_state = IDLE;
                    endcase
                end
            end
        endcase
    end
    
    // Output logic
    always_comb begin
        // Default outputs
        input_fifo_ready_o = 1'b0;
        deform_data_o = '0;
        deform_valid_o = 1'b0;
        cull_data_o = '0;
        cull_valid_o = 1'b0;
        deform_to_cull_ready_o = 1'b0;
        cull_to_deform_ready_o = 1'b0;
        
        case (state)
            IDLE: begin
                input_fifo_ready_o = 1'b1;
            end
            
            ROUTE_STATIC: begin
                // Static: bypass deformation, route directly to culling
                cull_data_o = gaussian_in;
                cull_valid_o = input_fifo_valid_i;
                input_fifo_ready_o = cull_ready_i;
            end
            
            ROUTE_DYNAMIC: begin
                // Dynamic: route to deformation first
                deform_data_o = gaussian_in;
                deform_valid_o = input_fifo_valid_i;
                input_fifo_ready_o = deform_ready_i;
                
                // Forward deformed output to culling
                deform_to_cull_ready_o = cull_ready_i;
            end
            
            ROUTE_QUASI_STATIC: begin
                // Quasi-static: route to culling first
                cull_data_o = gaussian_in;
                cull_valid_o = input_fifo_valid_i;
                input_fifo_ready_o = cull_ready_i;
                
                // Forward visible primitives to deformation
                cull_to_deform_ready_o = deform_ready_i;
            end
        endcase
    end
    
    // Forward deformed data to culling (for dynamic path)
    assign cull_data_o = (state == ROUTE_DYNAMIC && deform_to_cull_valid_i) ? 
                         deform_to_cull_data_i : 
                         ((state == ROUTE_STATIC || state == ROUTE_QUASI_STATIC) ? gaussian_in : '0);
    
    assign cull_valid_o = (state == ROUTE_DYNAMIC && deform_to_cull_valid_i) ? 
                         1'b1 : 
                         ((state == ROUTE_STATIC || state == ROUTE_QUASI_STATIC) ? input_fifo_valid_i : 1'b0);

endmodule
