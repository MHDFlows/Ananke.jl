#module ComputeFlux

# Flux Integrator
function ComputeFlux!(prob; Order=1)
    
    grid = prob.grid
    EOS  = prob.EOS
    tmp  = prob.tmp

    # get the flux vars
    w =  prob.sol.W
    F = prob.flux.F
    G = prob.flux.G
    H = prob.flux.H
    # Get back the infromation of all dimonions
    x1 = grid.x1
    x2 = grid.x2
    x3 = grid.x3
    is,ie = x1.is::Int,x1.ie::Int
    js,je = x2.js::Int,x2.je::Int
    ks,ke = x3.ks::Int,x3.ke::Int

    IDN          = grid.ind.ρ::Int
    IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int
    
    # Declare the upwind and downwind var
    wl = prob.tmp.wl
    wr = prob.tmp.wr

    @. wl *= 0
    @. wr *= 0
    # X₁ direction 
    if grid.nx::Int > 1
      if Order == 1
        ReconstructionFunction! = prob.equation.DonorCellX₁!
      else
        ReconstructionFunction! = prob.equation.ReconstructionFunction_X₁!
      end

      @timeit_debug prob.debugTimer "Reconstruct dir x" CUDA.@sync  begin
        ReconstructionFunction!(w, wl ,wr, grid)
      end
      @timeit_debug prob.debugTimer "Solver dir x" CUDA.@sync begin
        prob.equation.RoeSolver!(F, wl, wr, IVX, 
                                is, ie+1, js, je, ks, ke, 
                                EOS, grid, tmp)
      end
    end

    @. wl*=0
    @. wr*=0
    # X₂ direction
    if grid.ny::Int > 1
      if Order == 1
          ReconstructionFunction! = prob.equation.DonorCellX₂!
      else
          ReconstructionFunction! = prob.equation.ReconstructionFunction_X₂!
      end

      #if (B_field...)
      @timeit_debug prob.debugTimer "Reconstruct dir y" CUDA.@sync begin
        ReconstructionFunction!(w, wl ,wr, grid)
      end

      @timeit_debug prob.debugTimer "Solver dir y" CUDA.@sync begin
        prob.equation.RoeSolver!(G, wl, wr, IVY, 
                                 is, ie, js, je+1, ks, ke, 
                                 EOS, grid, tmp)
      end
    end


    @. wl  *= 0
    @. wr  *= 0    
    # X₃ direction 
    if grid.nz::Int > 1
      if Order == 1
          ReconstructionFunction! = prob.equation.DonorCellX₃!
      else
          ReconstructionFunction! = prob.equation.ReconstructionFunction_X₃!
      end

      @timeit_debug prob.debugTimer "Reconstruct dir z" CUDA.@sync  begin
        ReconstructionFunction!(w, wl ,wr, grid)
      end

      @timeit_debug prob.debugTimer "Solver dir z" CUDA.@sync  begin
        prob.equation.RoeSolver!(H, wl, wr, IVZ, 
                                 is, ie, js, je, ks, ke+1, 
                                 EOS, grid, tmp)
      end
    end
  return nothing 
end

#end