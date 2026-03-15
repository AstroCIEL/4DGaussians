// FRE Blending Core
// 8x8 array of blending units with 4 parallel lanes each

`include "gs_types.sv"
import gs_types::*;

module fre_bc (
    input  logic clk,
    input  logic rst_n,
    
    // Task input from WBS
    input  wbs_task_t                       task_i,
    input  logic                            task_valid_i,
    output logic                            task_ready_o,
    
    // Sorted Gaussian chunks from HSE
    input  sorted_chunk_t                   sorted_chunk_i,
    input  logic                            sorted_chunk_valid_i,
    output logic                            sorted_chunk_ready_o,
    
    // Gaussian data input (from external memory)
    input  deformed_gaussian_t              gaussian_data_i,
    input  logic                            gaussian_valid_i,
    output logic                            gaussian_ready_o,
    
    // Output pixels to interpolation unit
    output pixel_data_t                     pixel_data_o [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1],
    output logic                            pixel_valid_o,
    input  logic                            pixel_ready_i,
    
    // Core status
    output logic                            core_ready_o
);

    // Local SRAM Feature Buffer
    deformed_gaussian_t feature_buffer [0:1023];  // Max Gaussians per tile
    logic [GAUSSIAN_ID_WIDTH-1:0] buffer_write_ptr;
    logic [GAUSSIAN_ID_WIDTH-1:0] buffer_read_ptr;
    logic [GAUSSIAN_ID_WIDTH-1:0] num_gaussians;
    
    // Blending state machine
    typedef enum logic [2:0] {
        IDLE,
        LOAD_FEATURES,
        BLEND_PIXELS,
        OUTPUT_PIXELS
    } blend_state_e;
    
    blend_state_e state, next_state;
    
    // Pixel accumulation buffers (8x8 pixels)
    pixel_data_t pixel_buffer [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [BLEND_UNIT_SIZE-1:0][BLEND_UNIT_SIZE-1:0] pixel_valid_flags;
    
    // Alpha blending parameters
    logic [ALPHA_WIDTH-1:0] alpha [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [31:0] transmittance [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [COLOR_WIDTH-1:0] color_r [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [COLOR_WIDTH-1:0] color_g [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    logic [COLOR_WIDTH-1:0] color_b [0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1];
    
    // Gaussian processing index
    logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_idx;
    logic [GAUSSIAN_ID_WIDTH-1:0] chunk_idx;
    logic [3:0] gaussian_in_chunk_idx;
    
    // Temporary variables for alpha blending computation
    logic [ALPHA_WIDTH-1:0] alpha_contrib;
    logic [31:0] new_transmittance;
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            buffer_write_ptr <= '0;
            buffer_read_ptr <= '0;
            num_gaussians <= '0;
            gaussian_idx <= '0;
            chunk_idx <= '0;
            gaussian_in_chunk_idx <= '0;
            
            for (integer i = 0; i < BLEND_UNIT_SIZE; i++) begin
                for (integer j = 0; j < BLEND_UNIT_SIZE; j++) begin
                    pixel_buffer[i][j] <= '0;
                    pixel_valid_flags[i][j] <= 1'b0;
                    transmittance[i][j] <= 32'h3F800000;  // 1.0 in float32
                end
            end
        end else begin
            state <= next_state;
            
            if (state == LOAD_FEATURES && gaussian_valid_i && gaussian_ready_o) begin
                feature_buffer[buffer_write_ptr] <= gaussian_data_i;
                buffer_write_ptr <= buffer_write_ptr + 1;
            end
            
            if (state == IDLE && task_valid_i) begin
                num_gaussians <= task_i.num_gaussians;
                buffer_write_ptr <= '0;
                buffer_read_ptr <= '0;
                gaussian_idx <= '0;
                chunk_idx <= '0;
            end
            
            if (state == BLEND_PIXELS) begin
                gaussian_in_chunk_idx <= gaussian_in_chunk_idx + 1;
                if (gaussian_in_chunk_idx >= sorted_chunk_i.num_valid - 1) begin
                    gaussian_in_chunk_idx <= '0;
                    chunk_idx <= chunk_idx + 1;
                end
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (task_valid_i) begin
                    next_state = LOAD_FEATURES;
                end
            end
            LOAD_FEATURES: begin
                if (buffer_write_ptr >= num_gaussians) begin
                    next_state = BLEND_PIXELS;
                end
            end
            BLEND_PIXELS: begin
                if (gaussian_idx >= num_gaussians) begin
                    next_state = OUTPUT_PIXELS;
                end
            end
            OUTPUT_PIXELS: begin
                if (pixel_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Alpha blending computation
    // For each Gaussian, compute contribution to each pixel in 8x8 tile
    always_ff @(posedge clk) begin
        if (state == BLEND_PIXELS && gaussian_idx < num_gaussians) begin
            automatic deformed_gaussian_t g = feature_buffer[gaussian_idx];
            
            // Simplified alpha blending per pixel
            // Real implementation would compute 2D Gaussian falloff
            for (integer i = 0; i < BLEND_UNIT_SIZE; i++) begin
                for (integer j = 0; j < BLEND_UNIT_SIZE; j++) begin
                    // Compute alpha contribution (simplified)
                    alpha_contrib = g.opacity;  // Simplified
                    
                    // Update transmittance
                    // NOTE: Avoid floating-point literal arithmetic in synthesis.
                    // Use a simple integer-domain approximation for transmittance update.
                    // This is a placeholder; real implementation should use FP/fixed-point math.
                    new_transmittance = transmittance[i][j] - {24'b0, alpha_contrib};
                    
                    // Early termination: if transmittance is very low, skip remaining Gaussians
                    if (new_transmittance > 32'h3DCCCCCD) begin  // > 0.1
                        transmittance[i][j] <= new_transmittance;
                        
                        // Blend color (simplified - would use SH coefficients)
                        pixel_buffer[i][j].r <= pixel_buffer[i][j].r + 
                                                (g.sh_coeffs[7:0] * alpha_contrib);
                        pixel_buffer[i][j].g <= pixel_buffer[i][j].g + 
                                                (g.sh_coeffs[15:8] * alpha_contrib);
                        pixel_buffer[i][j].b <= pixel_buffer[i][j].b + 
                                                (g.sh_coeffs[23:16] * alpha_contrib);
                        pixel_buffer[i][j].alpha <= pixel_buffer[i][j].alpha + alpha_contrib;
                        pixel_buffer[i][j].depth <= g.depth;
                    end
                end
            end
            
            gaussian_idx <= gaussian_idx + 1;
        end
    end
    
    // Output pixel data
    always_comb begin
        for (integer i = 0; i < BLEND_UNIT_SIZE; i++) begin
            for (integer j = 0; j < BLEND_UNIT_SIZE; j++) begin
                pixel_data_o[i][j] = pixel_buffer[i][j];
            end
        end
    end
    
    // Handshaking
    assign task_ready_o = (state == IDLE);
    assign sorted_chunk_ready_o = (state == BLEND_PIXELS);
    assign gaussian_ready_o = (state == LOAD_FEATURES);
    assign pixel_valid_o = (state == OUTPUT_PIXELS);
    assign core_ready_o = (state == IDLE);

endmodule
