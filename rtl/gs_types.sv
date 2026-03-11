// 4DGS Accelerator Type Definitions and Parameters
// Shared definitions for all modules

package gs_types;

    // ========== System Parameters ==========
    parameter int TILE_SIZE = 32;              // Tile dimension: 32x32 pixels
    parameter int SUBTILE_SIZE = 4;            // Sub-tile dimension: 4x4 pixels
    parameter int NUM_CORES = 16;              // Number of parallel cores in HSE/FRE
    parameter int IQ_WINDOW_SIZE = 32;         // Instruction Queue window size K=32
    parameter int BLEND_UNIT_SIZE = 8;         // Blending unit array: 8x8
    parameter int BLEND_LANES = 4;            // Parallel lanes per blending unit
    
    // ========== Data Width Parameters ==========
    parameter int DATA_WIDTH = 512;            // AXI data width (bits)
    parameter int ADDR_WIDTH = 64;             // Address width
    parameter int GAUSSIAN_DATA_WIDTH = 256;   // Gaussian primitive data width
    parameter int MOTION_TAG_WIDTH = 2;        // Motion tag width (2 bits)
    parameter int TILE_ID_WIDTH = 10;          // Tile ID width (for 32x32 tiles in 1024x1024 screen)
    parameter int GAUSSIAN_ID_WIDTH = 20;      // Gaussian ID width
    parameter int DEPTH_WIDTH = 24;            // Depth value width
    parameter int COLOR_WIDTH = 24;            // RGB color width (8 bits per channel)
    parameter int ALPHA_WIDTH = 8;             // Alpha/opacity width
    parameter int WEIGHT_WIDTH = 16;           // Neural network weight width
    parameter int FEATURE_WIDTH = 32;          // Feature vector width
    
    // ========== Precision Parameters ==========
    parameter int FIXED_POINT_WIDTH = 16;      // Fixed-point width
    parameter int FIXED_POINT_FRAC = 8;        // Fractional bits
    
    // ========== Motion Tag Encoding ==========
    typedef enum logic [1:0] {
        MOTION_STATIC      = 2'b00,  // Static: bypass deformation
        MOTION_QUASI_STATIC = 2'b01, // Quasi-static: cull first, then deform if visible
        MOTION_DYNAMIC     = 2'b10   // Dynamic: deform first, then cull
    } motion_tag_e;
    
    // ========== Foveated Downsampling Rates ==========
    typedef enum logic [1:0] {
        DOWNSAMPLE_1X = 2'b00,  // No downsampling
        DOWNSAMPLE_2X = 2'b01,  // 2x downsampling
        DOWNSAMPLE_4X = 2'b10   // 4x downsampling
    } downsample_rate_e;
    
    // ========== Gaussian Primitive Structure ==========
    typedef struct packed {
        logic [31:0] mu_x, mu_y, mu_z;        // Mean position (3D)
        logic [31:0] sigma_xx, sigma_xy, sigma_xz;  // Covariance matrix elements
        logic [31:0] sigma_yy, sigma_yz, sigma_zz;
        logic [7:0]  opacity;                  // Base opacity
        logic [95:0] sh_coeffs;                // Spherical harmonics coefficients (12 coeffs * 8 bits)
        logic [1:0]  motion_tag;               // Motion tag
        logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
    } gaussian_primitive_t;
    
    // ========== Deformed Gaussian Structure ==========
    typedef struct packed {
        logic [31:0] mu_x, mu_y, mu_z;        // Deformed mean position
        logic [31:0] sigma_xx, sigma_xy, sigma_xz;
        logic [31:0] sigma_yy, sigma_yz, sigma_zz;
        logic [7:0]  opacity;
        logic [95:0] sh_coeffs;
        logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_id;
        logic [DEPTH_WIDTH-1:0] depth;         // Camera-space depth
        logic        visible;                  // Frustum visibility flag
    } deformed_gaussian_t;
    
    // ========== Tile Workload Structure ==========
    typedef struct packed {
        deformed_gaussian_t gaussian;
        logic [TILE_ID_WIDTH-1:0] tile_id;
        logic [15:0] workload_estimate;       // Estimated workload for this Gaussian
    } tile_workload_t;
    
    // ========== Task Structure for WBS ==========
    typedef struct packed {
        logic [TILE_ID_WIDTH-1:0] tile_id;
        logic [15:0] workload;
        logic [1:0]  downsample_rate;
        logic [GAUSSIAN_ID_WIDTH-1:0] start_gaussian_id;
        logic [GAUSSIAN_ID_WIDTH-1:0] num_gaussians;
    } wbs_task_t;
    
    // ========== Sorted Index Chunk ==========
    typedef struct {
        logic [GAUSSIAN_ID_WIDTH-1:0] gaussian_ids [0:15];  // Sorted Gaussian IDs
        logic [TILE_ID_WIDTH-1:0] tile_id;
        logic [4:0] num_valid;                // Number of valid IDs in this chunk
    } sorted_chunk_t;
    
    // ========== Pixel Data Structure ==========
    typedef struct packed {
        logic [7:0] r, g, b;                  // RGB channels
        logic [7:0] alpha;                    // Alpha channel
        logic [DEPTH_WIDTH-1:0] depth;
    } pixel_data_t;
    
    // ========== Camera Parameters ==========
    typedef struct {
        logic [31:0] view_matrix [0:15];      // 4x4 view matrix
        logic [31:0] proj_matrix [0:15];      // 4x4 projection matrix
        logic [31:0] time_step;                // Current time step t
    } camera_params_t;
    
    // ========== Gaze Parameters ==========
    typedef struct packed {
        logic [15:0] gaze_x, gaze_y;          // Gaze point in screen coordinates
        logic [15:0] fovea_radius;            // Fovea radius
    } gaze_params_t;
    
endpackage
