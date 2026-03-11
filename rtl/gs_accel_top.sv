// 4DGS Accelerator Top Module
// Top-level wrapper with AXI4 interfaces

`include "gs_types.sv"
import gs_types::*;

module gs_accel_top (
    input  logic clk,
    input  logic rst_n,
    
    // ========== AXI4 Master Interface (Read/Write to DRAM) ==========
    // Write Address Channel
    output logic [ADDR_WIDTH-1:0]           m_axi_awaddr,
    output logic [7:0]                      m_axi_awlen,
    output logic [2:0]                      m_axi_awsize,
    output logic [1:0]                      m_axi_awburst,
    output logic                            m_axi_awvalid,
    input  logic                            m_axi_awready,
    
    // Write Data Channel
    output logic [DATA_WIDTH-1:0]           m_axi_wdata,
    output logic [DATA_WIDTH/8-1:0]        m_axi_wstrb,
    output logic                            m_axi_wlast,
    output logic                            m_axi_wvalid,
    input  logic                            m_axi_wready,
    
    // Write Response Channel
    input  logic [1:0]                      m_axi_bresp,
    input  logic                            m_axi_bvalid,
    output logic                            m_axi_bready,
    
    // Read Address Channel
    output logic [ADDR_WIDTH-1:0]           m_axi_araddr,
    output logic [7:0]                      m_axi_arlen,
    output logic [2:0]                      m_axi_arsize,
    output logic [1:0]                      m_axi_arburst,
    output logic                            m_axi_arvalid,
    input  logic                            m_axi_arready,
    
    // Read Data Channel
    input  logic [DATA_WIDTH-1:0]           m_axi_rdata,
    input  logic [1:0]                      m_axi_rresp,
    input  logic                            m_axi_rlast,
    input  logic                            m_axi_rvalid,
    output logic                            m_axi_rready,
    
    // ========== AXI4-Lite Slave Interface (CSR) ==========
    // Write Address Channel
    input  logic [31:0]                     s_axi_awaddr,
    input  logic                            s_axi_awvalid,
    output logic                            s_axi_awready,
    
    // Write Data Channel
    input  logic [31:0]                     s_axi_wdata,
    input  logic [3:0]                      s_axi_wstrb,
    input  logic                            s_axi_wvalid,
    output logic                            s_axi_wready,
    
    // Write Response Channel
    output logic [1:0]                      s_axi_bresp,
    output logic                            s_axi_bvalid,
    input  logic                            s_axi_bready,
    
    // Read Address Channel
    input  logic [31:0]                     s_axi_araddr,
    input  logic                            s_axi_arvalid,
    output logic                            s_axi_arready,
    
    // Read Data Channel
    output logic [31:0]                    s_axi_rdata,
    output logic [1:0]                      s_axi_rresp,
    output logic                            s_axi_rvalid,
    input  logic                            s_axi_rready
);

    // ========== CSR Registers ==========
    camera_params_t camera_params;
    gaze_params_t gaze_params;
    logic [15:0] screen_width, screen_height;
    logic [ADDR_WIDTH-1:0] gaussian_data_base_addr;
    logic [ADDR_WIDTH-1:0] frame_buffer_base_addr;
    logic start_render;
    logic render_done;
    
    // CSR register map (simplified)
    // 0x00-0x3F: Camera view matrix (16 x 32-bit)
    // 0x40-0x4F: Camera projection matrix (16 x 32-bit)
    // 0x50: Time step (32-bit)
    // 0x54-0x57: Gaze parameters (gaze_x, gaze_y, fovea_radius)
    // 0x58-0x59: Screen resolution (width, height)
    // 0x5C: Gaussian data base address (64-bit, split into 2 registers)
    // 0x60: Frame buffer base address (64-bit, split into 2 registers)
    // 0x64: Control register (start_render bit)
    // 0x68: Status register (render_done bit)
    
    // AXI4-Lite CSR interface (simplified implementation)
    logic [31:0] csr_rdata;
    logic csr_write;
    logic [31:0] csr_waddr, csr_wdata;
    
    // Write handling
    assign s_axi_awready = s_axi_awvalid && s_axi_wvalid;
    assign s_axi_wready = s_axi_awvalid && s_axi_wvalid;
    assign csr_write = s_axi_awvalid && s_axi_wvalid;
    assign csr_waddr = s_axi_awaddr;
    assign csr_wdata = s_axi_wdata;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            camera_params <= '0;
            gaze_params <= '0;
            screen_width <= 1920;
            screen_height <= 1080;
            gaussian_data_base_addr <= '0;
            frame_buffer_base_addr <= '0;
            start_render <= 1'b0;
        end else begin
            if (csr_write) begin
                case (csr_waddr[7:2])
                    6'h00: camera_params.view_matrix[0] <= csr_wdata;
                    6'h01: camera_params.view_matrix[1] <= csr_wdata;
                    // ... more view matrix registers
                    6'h10: camera_params.proj_matrix[0] <= csr_wdata;
                    // ... more projection matrix registers
                    6'h14: camera_params.time_step <= csr_wdata;
                    6'h15: gaze_params.gaze_x <= csr_wdata[15:0];
                    6'h16: gaze_params.gaze_y <= csr_wdata[15:0];
                    6'h17: gaze_params.fovea_radius <= csr_wdata[15:0];
                    6'h18: screen_width <= csr_wdata[15:0];
                    6'h19: screen_height <= csr_wdata[15:0];
                    6'h1B: gaussian_data_base_addr[31:0] <= csr_wdata;
                    6'h1C: gaussian_data_base_addr[63:32] <= csr_wdata;
                    6'h1D: frame_buffer_base_addr[31:0] <= csr_wdata;
                    6'h1E: frame_buffer_base_addr[63:32] <= csr_wdata;
                    6'h19: start_render <= csr_wdata[0];
                endcase
            end
            if (render_done) begin
                start_render <= 1'b0;
            end
        end
    end
    
    // Read handling
    assign s_axi_arready = s_axi_arvalid && !s_axi_rvalid;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            csr_rdata <= '0;
            s_axi_rvalid <= 1'b0;
        end else begin
            if (s_axi_arvalid && s_axi_arready) begin
                case (s_axi_araddr[7:2])
                    6'h00: csr_rdata <= camera_params.view_matrix[0];
                    // ... more read cases
                    6'h1A: csr_rdata <= {16'h0, render_done, 15'h0};
                    default: csr_rdata <= 32'h0;
                endcase
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rready && s_axi_rvalid) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
    assign s_axi_rdata = csr_rdata;
    assign s_axi_bresp = 2'b00;
    assign s_axi_bvalid = csr_write;
    assign s_axi_rresp = 2'b00;
    
    // ========== AXI4 Master Interface (Simplified) ==========
    // Input FIFO for Gaussian data
    logic [GAUSSIAN_DATA_WIDTH-1:0] input_fifo_data;
    logic input_fifo_valid;
    logic input_fifo_ready;
    
    // AXI read controller (simplified)
    logic [ADDR_WIDTH-1:0] read_addr;
    logic [7:0] read_burst_len;
    logic read_in_progress;
    
    assign m_axi_araddr = read_addr;
    assign m_axi_arlen = read_burst_len;
    assign m_axi_arsize = 3'b110;  // 64 bytes (512 bits)
    assign m_axi_arburst = 2'b01;   // INCR
    assign m_axi_arvalid = start_render && !read_in_progress;
    assign m_axi_rready = input_fifo_ready;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_addr <= gaussian_data_base_addr;
            read_in_progress <= 1'b0;
            input_fifo_valid <= 1'b0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                read_in_progress <= 1'b1;
            end
            
            if (m_axi_rvalid && m_axi_rready) begin
                input_fifo_data <= m_axi_rdata[GAUSSIAN_DATA_WIDTH-1:0];
                input_fifo_valid <= 1'b1;
                
                if (m_axi_rlast) begin
                    read_in_progress <= 1'b0;
                    read_addr <= read_addr + (read_burst_len + 1) * DATA_WIDTH / 8;
                end
            end else begin
                input_fifo_valid <= 1'b0;
            end
        end
    end
    
    // AXI write controller for frame buffer (simplified)
    logic [ADDR_WIDTH-1:0] write_addr;
    logic write_in_progress;
    
    assign m_axi_awaddr = write_addr;
    assign m_axi_awlen = 7'h0;  // Single beat
    assign m_axi_awsize = 3'b010;  // 4 bytes
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = pixel_valid_o && !write_in_progress;
    
    assign m_axi_wdata = {8'h0, pixel_data_o.b, pixel_data_o.g, pixel_data_o.r};
    assign m_axi_wstrb = 4'hF;
    assign m_axi_wlast = 1'b1;
    assign m_axi_wvalid = pixel_valid_o && write_in_progress;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_addr <= frame_buffer_base_addr;
            write_in_progress <= 1'b0;
        end else begin
            if (m_axi_awvalid && m_axi_awready) begin
                write_in_progress <= 1'b1;
            end
            
            if (m_axi_wvalid && m_axi_wready) begin
                write_addr <= write_addr + 4;
                write_in_progress <= 1'b0;
            end
        end
    end
    
    assign m_axi_bready = 1'b1;
    
    // ========== Internal Engine Connections ==========
    // UDPE outputs
    tile_workload_t udpe_to_wbs;
    logic udpe_to_wbs_valid;
    logic udpe_to_wbs_ready;
    
    // WBS outputs
    wbs_task_t wbs_to_hse [0:NUM_CORES-1];
    logic [NUM_CORES-1:0] wbs_to_hse_valid;
    logic [NUM_CORES-1:0] wbs_to_hse_ready;
    
    wbs_task_t wbs_to_fre [0:NUM_CORES-1];
    logic [NUM_CORES-1:0] wbs_to_fre_valid;
    logic [NUM_CORES-1:0] wbs_to_fre_ready;
    
    // HSE outputs
    sorted_chunk_t hse_to_fre [0:NUM_CORES-1];
    logic [NUM_CORES-1:0] hse_to_fre_valid;
    logic [NUM_CORES-1:0] hse_to_fre_ready;
    
    // HSE/FRE core status
    logic [NUM_CORES-1:0] hse_core_ready;
    logic [NUM_CORES-1:0] fre_core_ready;
    
    // Gaussian data to HSE cores (from AXI)
    deformed_gaussian_t gaussian_to_hse [0:NUM_CORES-1];
    logic [NUM_CORES-1:0] gaussian_to_hse_valid;
    logic [NUM_CORES-1:0] gaussian_to_hse_ready;
    
    // FRE outputs
    pixel_data_t fre_pixel_out;
    logic [15:0] fre_pixel_x, fre_pixel_y;
    logic fre_pixel_valid;
    logic fre_pixel_ready;
    
    // Weight cache interface (simplified - would connect to external cache)
    logic [31:0] weight_addr;
    logic weight_req;
    logic [WEIGHT_WIDTH-1:0] weight_data [0:7];
    logic weight_valid;
    
    // ========== Instantiate UDPE ==========
    udpe_top u_udpe (
        .clk(clk),
        .rst_n(rst_n),
        .input_fifo_data_i(input_fifo_data),
        .input_fifo_valid_i(input_fifo_valid),
        .input_fifo_ready_o(input_fifo_ready),
        .camera_params_i(camera_params),
        .screen_width_i(screen_width),
        .screen_height_i(screen_height),
        .weight_addr_o(weight_addr),
        .weight_req_o(weight_req),
        .weight_data_i(weight_data),
        .weight_valid_i(weight_valid),
        .tile_workload_o(udpe_to_wbs),
        .tile_workload_valid_o(udpe_to_wbs_valid),
        .tile_workload_ready_i(udpe_to_wbs_ready)
    );
    
    // ========== Instantiate WBS ==========
    wbs_top u_wbs (
        .clk(clk),
        .rst_n(rst_n),
        .tile_workload_i(udpe_to_wbs),
        .tile_workload_valid_i(udpe_to_wbs_valid),
        .tile_workload_ready_o(udpe_to_wbs_ready),
        .gaze_params_i(gaze_params),
        .screen_width_i(screen_width),
        .screen_height_i(screen_height),
        .hse_core_ready_i(hse_core_ready),
        .fre_core_ready_i(fre_core_ready),
        .hse_task_o(wbs_to_hse),
        .hse_task_valid_o(wbs_to_hse_valid),
        .hse_task_ready_i(wbs_to_hse_ready),
        .fre_task_o(wbs_to_fre),
        .fre_task_valid_o(wbs_to_fre_valid),
        .fre_task_ready_i(wbs_to_fre_ready)
    );
    
    // ========== Instantiate HSE ==========
    hse_top u_hse (
        .clk(clk),
        .rst_n(rst_n),
        .task_i(wbs_to_hse),
        .task_valid_i(wbs_to_hse_valid),
        .task_ready_o(wbs_to_hse_ready),
        .gaussian_data_i(gaussian_to_hse),
        .gaussian_valid_i(gaussian_to_hse_valid),
        .gaussian_ready_o(gaussian_to_hse_ready),
        .sorted_chunk_o(hse_to_fre),
        .sorted_chunk_valid_o(hse_to_fre_valid),
        .sorted_chunk_ready_i(hse_to_fre_ready),
        .core_ready_o(hse_core_ready)
    );
    
    // ========== Instantiate FRE ==========
    fre_top u_fre (
        .clk(clk),
        .rst_n(rst_n),
        .task_i(wbs_to_fre),
        .task_valid_i(wbs_to_fre_valid),
        .task_ready_o(wbs_to_fre_ready),
        .sorted_chunk_i(hse_to_fre),
        .sorted_chunk_valid_i(hse_to_fre_valid),
        .sorted_chunk_ready_o(hse_to_fre_ready),
        .gaussian_data_i(gaussian_to_hse),
        .gaussian_valid_i(gaussian_to_hse_valid),
        .gaussian_ready_o(gaussian_to_hse_ready),
        .pixel_data_o(fre_pixel_out),
        .pixel_x_o(fre_pixel_x),
        .pixel_y_o(fre_pixel_y),
        .pixel_valid_o(fre_pixel_valid),
        .pixel_ready_i(fre_pixel_ready),
        .core_ready_o(fre_core_ready)
    );
    
    // Connect FRE outputs to top-level signals
    assign pixel_data_o = fre_pixel_out;
    assign pixel_x_o = fre_pixel_x;
    assign pixel_y_o = fre_pixel_y;
    assign pixel_valid_o = fre_pixel_valid;
    assign fre_pixel_ready = pixel_ready_i;
    
    // Render done detection (simplified)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            render_done <= 1'b0;
        end else begin
            // Simplified: render done when all cores are idle and no more data
            render_done <= (&hse_core_ready) && (&fre_core_ready) && 
                          !input_fifo_valid && !udpe_to_wbs_valid;
        end
    end
    
    // Note: Gaussian data distribution to HSE cores would be handled by
    // a separate data distributor module based on task assignments
    // This is simplified - in real implementation, would route based on tile_id

endmodule
