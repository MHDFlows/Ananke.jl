#module TimeIntegrator

include("hydro/FluxIntegrator.jl");
function VL2!(prob) 
    
    # Half time step Δt/2 for F^{n+1/2}
    # Every New Time Step
    # U (copy) -> U_half
    #  Update U_half from U + dFdx
    
    #Full Step for updating U
    # Update W from U_half
    # Update F_half from W_half
    # update U from dF_half dx
    # exchange boundary value
    
    dt = prob.clock.dt;
    grid = prob.grid;
    
    U = prob.sol.U;
    W = prob.sol.W;
    U_half = prob.sol.U_half;
    BoundaryExchangeFunction! = prob.equation.BoundaryExchangeFunction!;
    
    #Sync U_half and U
    copyto!(U_half, U);

    # Half Step First (dt/2 with order = 1) then Full step (dt with order = 2)
    for (Uᵢ, order, dtᵢ) ∈ zip((U_half,U),(1, prob.flag.SpatialOrder),(dt/2,dt)) 
      @timeit_debug prob.debugTimer "Flux intregation" CUDA.@sync begin
        FluxIntegrator!(Uᵢ, dtᵢ, prob; Order=order); 
      end

      @timeit_debug prob.debugTimer "Cons to Prims" CUDA.@sync begin
        prob.EOS.ConservedToPrimitive!(W, Uᵢ, prob);
        #@show sum(isnan.(W))
      end

      @timeit_debug prob.debugTimer "Boundary Exchange" CUDA.@sync begin
        BoundaryExchangeFunction!(W, grid);
      end
    end

    return nothing
end

#end