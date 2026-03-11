// FRE Interpolation Reconstructor
// Bilinear interpolation for foveated downsampling

`include "gs_types.sv"
import gs_types::*;

module fre_interp (
    input  logic clk,
    input  logic rst_n,
    
    // Input pixels from blending cores (16 cores, each 8x8)
    input  pixel_data_t                     pixel_data_i [0:NUM_CORES-1][0:BLEND_UNIT_SIZE-1][0:BLEND_UNIT_SIZE-1],
    input  logic [NUM_CORES-1:0]             pixel_valid_i,
    output logic [NUM_CORES-1:0]             pixel_ready_o,
    
    // Downsampling rates from tasks
    input  downsample_rate_e                 downsample_rate_i [0:NUM_CORES-1],
    
    // Tile coordinates for boundary caching
    input  logic [TILE_ID_WIDTH-1:0]        tile_id_i [0:NUM_CORES-1],
    
    // Output to frame buffer
    output pixel_data_t                     pixel_data_o,
    output logic [15:0]                     pixel_x_o,
    output logic [15:0]                     pixel_y_o,
    output logic                            pixel_valid_o,
    input  logic                            pixel_ready_i
);

    // Boundary pixel cache (for cross-tile seam elimination)
    pixel_data_t boundary_cache_h [0:1023];  // Horizontal boundaries
    pixel_data_t boundary_cache_v [0:1023];  // Vertical boundaries
    logic [TILE_ID_WIDTH-1:0] cached_tile_id_h [0:1023];
    logic [TILE_ID_WIDTH-1:0] cached_tile_id_v [0:1023];
    
    // Interpolation state machine
    typedef enum logic [1:0] {
        IDLE,
        INTERPOLATE,
        OUTPUT
    } interp_state_e;
    
    interp_state_e state, next_state;
    logic [3:0] current_core;
    logic [2:0] pixel_x, pixel_y;  // Within 8x8 block
    
    // Bilinear interpolation
    function automatic pixel_data_t bilinear_interp(
        input pixel_data_t p00, p01, p10, p11,
        input logic [7:0] fx, fy  // Fractional parts (0-255)
    );
        pixel_data_t result;
        logic [15:0] r00, r01, r10, r11;
        logic [15:0] g00, g01, g10, g11;
        logic [15:0] b00, b01, b10, b11;
        
        // Interpolate RGB
        r00 = p00.r; r01 = p01.r; r10 = p10.r; r11 = p11.r;
        g00 = p00.g; g01 = p01.g; g10 = p10.g; g11 = p11.g;
        b00 = p00.b; b01 = p01.b; b10 = p10.b; b11 = p11.b;
        
        result.r = ((r00 * (256 - fx) + r01 * fx) * (256 - fy) + 
                    (r10 * (256 - fx) + r11 * fx) * fy) / 65536;
        result.g = ((g00 * (256 - fx) + g01 * fx) * (256 - fy) + 
                    (g10 * (256 - fx) + g11 * fx) * fy) / 65536;
        result.b = ((b00 * (256 - fx) + b01 * fx) * (256 - fy) + 
                    (b10 * (256 - fx) + b11 * fx) * fy) / 65536;
        
        result.alpha = p00.alpha;  // Use first pixel's alpha
        result.depth = p00.depth;   // Use first pixel's depth
        
        return result;
    endfunction
    
    // State register
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            current_core <= '0;
            pixel_x <= '0;
            pixel_y <= '0;
        end else begin
            state <= next_state;
            
            if (state == INTERPOLATE) begin
                pixel_x <= pixel_x + 1;
                if (pixel_x >= BLEND_UNIT_SIZE - 1) begin
                    pixel_x <= '0;
                    pixel_y <= pixel_y + 1;
                    if (pixel_y >= BLEND_UNIT_SIZE - 1) begin
                        pixel_y <= '0;
                        current_core <= current_core + 1;
                    end
                end
            end
        end
    end
    
    // Next state logic
    always_comb begin
        next_state = state;
        case (state)
            IDLE: begin
                if (|pixel_valid_i) begin
                    next_state = INTERPOLATE;
                end
            end
            INTERPOLATE: begin
                if (current_core >= NUM_CORES && pixel_x == BLEND_UNIT_SIZE - 1 && 
                    pixel_y == BLEND_UNIT_SIZE - 1) begin
                    next_state = OUTPUT;
                end
            end
            OUTPUT: begin
                if (pixel_ready_i) begin
                    next_state = IDLE;
                end
            end
        endcase
    end
    
    // Interpolation logic based on downsampling rate
    pixel_data_t interpolated_pixel;
    logic [15:0] output_x, output_y;
    
    always_comb begin
        downsample_rate_e rate = downsample_rate_i[current_core];
        pixel_data_t p00, p01, p10, p11;
        
        // Get neighboring pixels for interpolation
        p00 = pixel_data_i[current_core][pixel_y][pixel_x];
        p01 = (pixel_x < BLEND_UNIT_SIZE - 1) ? 
              pixel_data_i[current_core][pixel_y][pixel_x + 1] : p00;
        p10 = (pixel_y < BLEND_UNIT_SIZE - 1) ? 
              pixel_data_i[current_core][pixel_y + 1][pixel_x] : p00;
        p11 = (pixel_x < BLEND_UNIT_SIZE - 1 && pixel_y < BLEND_UNIT_SIZE - 1) ? 
              pixel_data_i[current_core][pixel_y + 1][pixel_x + 1] : p00;
        
        case (rate)
            DOWNSAMPLE_1X: begin
                // No interpolation needed
                interpolated_pixel = p00;
            end
            DOWNSAMPLE_2X: begin
                // Interpolate 2x2 -> 1 pixel
                interpolated_pixel = bilinear_interp(p00, p01, p10, p11, 128, 128);
            end
            DOWNSAMPLE_4X: begin
                // Interpolate 4x4 -> 1 pixel
                interpolated_pixel = bilinear_interp(p00, p01, p10, p11, 128, 128);
            end
            default: begin
                interpolated_pixel = p00;
            end
        endcase
        
        // Compute output coordinates
        logic [4:0] tile_x, tile_y;
        tile_x = tile_id_i[current_core] % 32;  // Assuming 32 tiles per row
        tile_y = tile_id_i[current_core] / 32;
        output_x = tile_x * TILE_SIZE + pixel_x;
        output_y = tile_y * TILE_SIZE + pixel_y;
    end
    
    // Output assignment
    always_ff @(posedge clk) begin
        if (state == INTERPOLATE) begin
            pixel_data_o <= interpolated_pixel;
            pixel_x_o <= output_x;
            pixel_y_o <= output_y;
        end
    end
    
    // Handshaking
    assign pixel_ready_o = (state == IDLE);
    assign pixel_valid_o = (state == OUTPUT);

endmodule
