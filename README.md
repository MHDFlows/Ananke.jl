# Project-Ananke

Project Anake is a on-onging CFD project, a highly efficient Magnetohydrodynamics (MHD) direct fluid GPU based code implemented on high level programming language julia. It is inspired by [`Athena++`](https://github.com/PrincetonUniversity/athena)/[`MHDFLows.jl`](https://github.com/MHDFlows/MHDFlows.jl)/[`FourierFlows.jl`](https://github.com/FourierFlows/FourierFlows.jl). Its goals is to achieve a native GPU based, highly parallel, and efficient MHD simulation code using Finite Volume Method(FVM).

At the end of this proejct, we aim to provide a prototype MHD code, which will support the following features:

1. 1D/2D/3D uniform Grid simulation on cartesian coordinate system
2. Hydrodynamics/Magnetohydrodynamics Equation Solver
3. Support Isothermal/Adiabatic equation of state.  

# Current State of development: 
Version  : v.0.1.0  
Phase  I : Finished & under testing  
Phase II : Under development  

To achieve those features, we split our development into three phases:  
## I) Simple GPU HD simulation with HLLC/HLLE Solver supporting  uniform cartesian coordinate
    - Data structure & Problem construction 
    - Recontruction construction  (DC(1)/ PLM(2)/ PPM(3) )
    - Solver Construction  ( HLLC for addiabatic/ HLLE for isothermal)
    - Basic Integrator Construction  (second-order accurate van Leer predictor-corrector scheme)
    - Boundary value problem constructor  (periodic/outflow/reflective)
## II) MPI implementation  & Curvilinear coordinate Support
    - MPI Meshblock distribution
    - MPI Boundary value exchange function
    - Problem diagnotics for global statistics after each iteration (e.g. Total Energy)   
    - Global FFT contructor  (with CuFFTMp)
    - Support for cylindrical/ spherical coordinates (optional)
## III) B-field implementation  
    - Reconstruction method with B-field 
    - HLLD Solvers  
    - Div B Free algorthm (CT)

# Compatiblility :
Support Julia v1.8.3 with CUDA compatible GPU

# Performance Evaluation:
Ananke is still under development and lack of optimzation. User should expect the perforamnce could change from time to time. As for now, Ananke is mainly memory-bound. So the performance decline moderately from swtiching between `Float32/Float64` calulation. Nonetheless, as the current test is based on high-end gaming card with nerfed VRAM/`FP64 Core` configuration, one could expect better performance for `Float64` calculation for data center GPU.  
3090 24GB with 199 iterations after warm up process
| Method (M Zones Update /s )  | $128^3$ | $256^3$ | $384^3$ |
| ---------------------------- | ------- | ------- | ------- |
| VL2 + PLM + HLLC(HD/Float32) | 86.278  | 192.651 | 242.94  |
| VL2 + PLM + HLLC(HD/Float64) | 60.918  | 157.583 | 186.07  |
| VL2 + PPM + HLLC(HD/Float32) | 74.528  | 130.43  | 155.41  | 
| VL2 + PPM + HLLC(HD/Float64) | 60.996  | 98.529  | 109.78  |

The performance comparsion to Athena++ (Table 3 in Stone et al. 2020)  
2× Skylake-SP Gold 614 (Total 40 Cores)

| Method (M Zones Update /s )  | about $256^3$  |
| ---------------------------- | -------------- |
| VL2 + PLM + HLLC(HD/Float64) |     84.769     | 
| VL2 + PPM + HLLC(HD/Float64) |     49.759     |

For breakdown of the runtime (i.e. `VL2 + PPM + HLLC (HD/Float32)` in below), as expected, Ananke spend most of the time in recontruction state & Flux construction step.

     ────────────────────────────────────────────────────────────────────────────────
                                            Time                    Allocations      
                                   ───────────────────────   ────────────────────────
           Tot / % measured:            91.9s /  79.3%           91.7MiB /  93.8%    
     Section               ncalls     time    %tot     avg     alloc    %tot      avg
     ────────────────────────────────────────────────────────────────────────────────
     Time Stepper             199    71.0s   97.4%   357ms   77.2MiB   89.8%   397KiB
       Flux intregation       398    67.0s   92.0%   168ms   34.7MiB   40.3%  89.2KiB
         Reconstruct dir x    398    14.0s   19.3%  35.3ms   4.15MiB    4.8%  10.7KiB
         Reconstruct dir y    398    13.6s   18.6%  34.0ms   4.15MiB    4.8%  10.7KiB
         Reconstruct dir z    398    13.5s   18.6%  34.0ms   4.15MiB    4.8%  10.7KiB
         Solver dir y         398    7.57s   10.4%  19.0ms   3.57MiB    4.1%  9.18KiB
         Solver dir z         398    7.56s   10.4%  19.0ms   3.57MiB    4.1%  9.18KiB
         Solver dir x         398    7.54s   10.3%  18.9ms   3.58MiB    4.2%  9.20KiB
         Adding ∂F∂x          398    3.14s    4.3%  7.88ms   6.38MiB    7.4%  16.4KiB
       Cons to Prims          398    2.74s    3.8%  6.89ms   15.9MiB   18.5%  41.0KiB
       Boundary Exchange      398    1.18s    1.6%  2.96ms   26.5MiB   30.8%  68.1KiB
     CFL                      199    1.91s    2.6%  9.60ms   8.82MiB   10.2%  45.4KiB
     User defined function    199    182μs    0.0%   915ns     0.00B    0.0%    0.00B
     ────────────────────────────────────────────────────────────────────────────────

# Example
The user interface of the Ananke is inherited from [`MHDFlows.jl`](https://github.com/MHDFlows/MHDFlows.jl) and they share most of the workflow.
[Few Examples ](https://github.com/MHDFlows/Ananke-Example)  were set to illustrate the workflow of performing 1D/2D/3D simulation and its visualization.