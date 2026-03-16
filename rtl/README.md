# Hardware Architecture Specification: 4DGS Accelerator

dedicated asic design of `xrh_iccad2026_0311.pdf`.

## 1. System Overview

This document defines the module boundaries, functionalities, and interfaces for the Real-Time 4D Gaussian Splatting (4DGS) Accelerator. The architecture is designed to optimize memory bandwidth and computational latency through Motion-Aware Tagging and Deformation Skipping (MTDS) , Windowed Spatial-Locality Aware Scheduling (WSLAS) , and Motion-aware Foveated Rendering (MAFR).

The top-level system acts as a memory-mapped peripheral via a standard bus interface (e.g., AXI4) reading from external LPDDR4 DRAM  and consists of four main pipelined stages:

1. **UDPE:** Unified Deformation-Preprocess Engine 
2. **WBS:** Workload Balancing Scheduler 
3. **HSE:** Hierarchical Sorting Engine 
4. **FRE:** Foveated Rasterizing Engine 

## 2. Top-Level System Interfaces (`gs_accel_top.sv`)

**Description:** The top-level wrapper that instantiates all four main engines and handles AXI4 transactions to main memory.

**Key Interfaces:**

* `clk`, `rst_n`: Global clock (target 1GHz) and active-low reset.
* `AXI_AW*`, `AXI_W*`, `AXI_B*`, `AXI_AR*`, `AXI_R*`: Standard AXI4 Master interface for reading Gaussian parameters and writing the final frame buffer.
* `CSR_IF`: AXI-Lite Slave interface for configuration (e.g., camera poses, gaze direction, time step $t$).

---

## 3. Module Specifications

### 3.1 Unified Deformation-Preprocess Engine (UDPE) (`udpe_top.sv`)

**Description:** Integrates deformation computation with preprocessing functions (frustum culling, 2D feature computation, and tile intersection testing) to maximize hardware parallelism.

**Sub-Modules to Implement:**

1. **Parsing Dispatcher (`udpe_dispatcher.sv`):**
* *Function:* Reads the 2-bit motion tags from the Input FIFO.

* *Logic:* Controls multiplexers/crossbars to route Gaussian primitives to either the Deformation Unit or directly to the Culling Unit.
* Tag `00` (Static): Bypass deformation entirely.
* Tag `01` (Quasi-static): Route to Culling first, then Deformation if visible.
* Tag `10` (Dynamic): Route to Deformation first, then Culling.

2. **Deformation Unit (`udpe_deform.sv`):**
* *Function:* Executes a lightweight neural network to update Gaussian parameters for the current frame.
* *Components:* Requires a Weight Cache, Systolic Array for MAC operations, and a Feature Buffer.

3. **Culling Unit (`udpe_culling.sv`):**
* *Function:* Calculates camera-space coordinates and evaluates frustum visibility flags.

* *Components:* Projection Unit, Culling Logic.

4. **2D Feature & Intersection Unit (`udpe_intersect.sv`):**
* *Function:* Computes screen-space features for visible primitives and outputs the updated Gaussian parameters and tile workload lists.

**Key Interfaces:**

* `input_fifo_data_i`, `input_fifo_valid_i`: Raw Gaussian data and 2-bit tags from DRAM.
* `tile_workload_data_o`, `tile_workload_valid_o`: Updated Gaussian properties and intersecting tile IDs passed to the WBS.

### 3.2 Workload Balancing Scheduler (WBS) (`wbs_top.sv`)

**Description:** Physically implements the WSLAS algorithm. It handles the severe tile load imbalance by mapping tasks to idle HSE/FRE cores.

**Sub-Modules to Implement:**

1. **Load Generator (`wbs_load_gen.sv`):**
* *Function:* Estimates computational workload based on the Gaussian count within a tile. Applies MAFR downsampling rates (1x, 2x, 4x) based on the tile's eccentricity from the user's gaze point. Outputs tasks following a Hilbert curve sequence.

2. **Instruction Queue (IQ) (`wbs_iq.sv`):**
* *Function:* A FIFO/Buffer that stores the active candidate window of size $K=32$.

3. **Tile Arbiter (`wbs_arbiter.sv`):**
* *Function:* A combinational comparator tree that rapidly identifies the tile with the maximum workload from the active window in the IQ.

4. **Task Dispatcher (`wbs_dispatcher.sv`):**
* *Function:* Monitors the ready/idle status of the 16 subcores in the HSE and FRE. Dispatches the heaviest tile from the Arbiter to an available core.

### 3.3 Hierarchical Sorting Engine (HSE) (`hse_top.sv`)

**Description:** Performs depth sorting of Gaussians within dispatched tiles.

**Sub-Modules to Implement:**

1. **HSE Core Array:** Instantiate 16 parallel sorting cores.

2. **HSE Core (`hse_core.sv`):**
* *Function:* Executes a two-stage sorting process.
* *Components:* Quick Sort Unit for coarse sorting and Bitonic Sort Unit for fine-grained sorting.
* *Interface:* Sends sorted index chunks to the corresponding core in the FRE.


### 3.4 Foveated Rasterizing Engine (FRE) (`fre_top.sv`)

**Description:** Executes the foveated $\alpha$-blending and outputs the final pixel values to the frame buffer.

**Sub-Modules to Implement:**

1. **Blending Core Array:** Instantiate 16 Blending Cores (BCs).

2. **Blending Core (BC) (`fre_bc.sv`):**
* *Components:* An $8\times8$ array of blending units.
* *Microarchitecture:* Each unit contains 4 parallel lanes for $\alpha$ calculation, transmittance updating (with early termination logic), and color blending for 4 individual pixels.
* *Memory:* Dedicated local SRAM Feature Buffer per core to store Gaussian attributes for the currently assigned tile.

3. **Interpolation Reconstructor (`fre_interp.sv`):**
* *Function:* Shared module that performs bilinear interpolation to reconstruct omitted pixels based on the foveated downsampling rate (1x, 2x, 4x).
* *Logic:* Caches boundary pixels of tiles to eliminate cross-tile seam artifacts during reconstruction before writing to the Output Buffer.

## 4. Implementation Directives for the Front-End Engineer

1. **Parameterization:** Make bus widths, precision (e.g., fixed-point vs float16), tile dimensions ($32\times32$), and sub-tile dimensions ($4\times4$) highly parameterized using SV `parameter` or `localparam` definitions.
2. **Ready/Valid Handshaking:** Use strict `valid`/`ready` handshaking protocols between all main engines (UDPE $\rightarrow$ WBS $\rightarrow$ HSE $\rightarrow$ FRE) to prevent data loss during stalls.
3. **Clock Gating:** Given the edge-scenario power budget, insert clock gating logic at the module level, especially for the 16 parallel cores in the HSE and FRE when they are waiting for the WBS to dispatch a task.
