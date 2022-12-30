function SetUpProblemIC!(prob;  ρ = [],  p = [], cₛ = 0.0,
                               ux = [], uy = [], uz =[],
                               bx = [], by = [], bz =[])

  grid = prob.grid
  IDN  = grid.ind.ρ::Int
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int;
  
  @views prim = prob.sol.W;
  @views  con = prob.sol.U;

  # Copy the data to both output and solution array
  for (primᵢ,indᵢ) in zip((ρ, ux, uy, uz),
                         (IDN,IVX,IVY,IVZ))
    CopyDataToProb!(view(prim,:, :, :, indᵢ), primᵢ)
  end

  movEOS!(prob, prim, p, cₛ)

  #= Template script from MHDFlows, b is not enable yet
  if prob.flag.b 
    for (bᵢ,prob_bᵢ,bᵢind) in zip([bx,by,bz],[vars.bx,vars.by,vars.bz],
                                  [params.bx_ind,params.by_ind,params.bz_ind])
      if bᵢ != []
        @views sol₀ =  sol[:, :, :, bᵢind];
        copyto!(prob_bᵢ,bᵢ);
      end
    end
  end
  =#

  prob.EOS.PrimitivetoConserved!(prob.sol.W,prob.sol.U,prob)
  prob.equation.BoundaryExchangeFunction!(prim,grid);
  prob.equation.BoundaryExchangeFunction!(con, grid);

  return nothing
end

function ConPrimCheck(a,b)
  if a != [] && b != []
    error("Con & Prim vars cannot be defined at one time!")
   end
end

function CopyDataToProb!(b,a)
  if a != [] 
    copyto!(b, a);
  end
end

function movEOS!(prob, prim, p, cₛ)
  if prob.EOS.EOSType == "Adiabatic" 
    cₛ != 0.0 ? error("User define both cₛ and pressure in Adiabatic EOS!") : nothing
    CopyDataToProb!(view(prim,:, :, :, prob.grid.ind.P), p)
  elseif prob.EOS.EOStype == "Isothermal"
    p == [] ? error("User define both cₛ and pressure in Isothermal EOS!") : nothing
    prob.EOS.cₛ = cₛ
  end
end