module Isothermal
using CUDA,Statistics
export ConservedToPrimitive!,
       PrimitivetoConserved!,
       UpdateCFL

@inline function ConservedToPrimitive!(prim,con,prob)
    
  grid = prob.grid;

  # Update [ρ,M1,M2,M3,e] -> [ρ,v1,v2,v3,P]
  x1,x2,x3 = grid.x1,grid.x2,grid.x3;
    
  # Declare the ind
  is,ie = x1.is::Int,x1.ie::Int;
  js,je = x2.js::Int,x2.je::Int;
  ks,ke = x3.ks::Int,x3.ke::Int;
    
  IDN          = grid.ind.ρ ::Int;
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int;
  IM1,IM2,IM3  = grid.ind.p₁::Int,grid.ind.p₂::Int,grid.ind.p₃::Int;
    
  rho_floor = 1e-15;
    
  ρc = view( con, is:ie ,js:je, ks:ke,IDN)
  KE = view(prob.tmp.wl,is:ie ,js:je ,ks:ke,IDN)
   ρ = view(prim, is:ie ,js:je ,ks:ke, IDN)
  iv,jv,kv = view(prim, is:ie ,js:je, ks:ke, IVX), view(prim, is:ie, js:je, ks:ke, IVY), view(prim, is:ie, js:je, ks:ke, IVZ)
  ip,jp,kp = view( con, is:ie ,js:je ,ks:ke, IM1), view( con, is:ie, js:je, ks:ke, IM2), view( con, is:ie, js:je, ks:ke, IM3)
  
  @.  ρ = ρc;
  @. iv = ip/ρ;
  @. jv = jp/ρ;
  @. kv = kp/ρ;
  
  return nothing;
end

function UpdateCFL!(prob,W; Coef=0.3, γ = 5/3)
  dt = 1e10
  square_maximum(A) =  mapreduce(x->x*x,max,A);
  #Get the data type
  x1 = prob.grid.x1;
  x2 = prob.grid.x2;
  x3 = prob.grid.x3;
  is,ie = x1.is::Int,x1.ie::Int;
  js,je = x2.js::Int,x2.je::Int;
  ks,ke = x3.ks::Int,x3.ke::Int;

  #Get the dx
  Δx₁ = x1.Δx1f[1];
  Δx₂ = x2.Δx2f[1];
  Δx₃ = x3.Δx3f[1];

  #Get the var index
  IDN          = prob.grid.ind.ρ ::Int
  IVX,IVY,IVZ  = prob.grid.ind.v₁::Int,prob.grid.ind.v₂::Int,prob.grid.ind.v₃::Int

  #Set up the tmp using wl
  iv = view(W, :, :, :, IVX)
  jv = view(W, :, :, :, IVY)
  kv = view(W, :, :, :, IVZ)
  #Maxmium velocity 
  vxmax = √(square_maximum(iv))
  vymax = √(square_maximum(jv))
  vzmax = √(square_maximum(kv))
  csmax = prob.EOS.cₛ
  dt = Coef*minimum((Δx₁/vxmax, Δx₂/vymax, Δx₃/vzmax, 
                     Δx₁/csmax, Δx₂/csmax, Δx₃/csmax,dt))
  
  if isnan(dt)
    error("detected NaN in CFL function!");
    end
  prob.clock.dt = dt;
  return nothing;

end



function PrimitivetoConserved!(prim,con,prob)

  grid = prob.grid;
  EOS  = prob.EOS;

  # Update [ρ,M1,M2,M3,e] -> [ρ,v1,v2,v3,P]
  x1,x2,x3 = grid.x1,grid.x2,grid.x3;
  
  # Declare the ind
  is,ie = x1.is::Int,x1.ie::Int;
  js,je = x2.js::Int,x2.je::Int;
  ks,ke = x3.ks::Int,x3.ke::Int;
  
  IDN,IEN,IPR  = grid.ind.ρ ::Int,grid.ind.e ::Int,grid.ind.P ::Int;
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int;
  IM1,IM2,IM3  = grid.ind.p₁::Int,grid.ind.p₂::Int,grid.ind.p₃::Int;
  
  γ = EOS.γ;
  γm1 = γ - 1.0;
  
  rho_floor = 1e-15;
  pre_floor = 1e-15;
  ρc =  view(con, is:ie ,js:je, ks:ke,IDN)
  KE = view(prob.tmp.wl,is:ie ,js:je ,ks:ke,IDN)
   ρ, p, e = view(prim, is:ie ,js:je ,ks:ke, IDN), view(prim, is:ie, js:je, ks:ke, IPR), view( con, is:ie, js:je, ks:ke, IEN)
  iv,jv,kv = view(prim, is:ie ,js:je, ks:ke, IVX), view(prim, is:ie, js:je, ks:ke, IVY), view(prim, is:ie, js:je, ks:ke, IVZ)
  ip,jp,kp = view( con, is:ie ,js:je ,ks:ke, IM1), view( con, is:ie, js:je, ks:ke, IM2), view( con, is:ie, js:je, ks:ke, IM3)

  @. ρc = ρ
  @. ip = ρ*iv
  @. jp = ρ*jv
  @. kp = ρ*kv
  @. KE = 0.5*ρ*(iv.^2 + jv.^2 + kv.^2)
  @.  e = p/γm1 + KE

  # correct the value if it is lower than floor value
  threads  = (1,32,32)
  blocks   = (1, ceil(Int,size(p,2)/threads[2]), ceil(Int,size(p,3)/threads[3]))
  @cuda blocks = blocks threads = threads Checkfloorvalue!(ρ, ρc, p, e, KE,
                                                           rho_floor, pre_floor, γm1,
                                                           is, ie, js, je, ks, ke)
  return nothing 
end
# correct the value if it is lower than floor value
function Checkfloorvalue!(ρ, ρc, p, e, KE,
                          ρ_fl, p_fl, γm1,
                          is, ie, js, je, ks, ke)

  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
 
  if j ∈ (js:je) && k ∈ (ks:ke) && i ∈ (is:ie)
    for i = is:ie  
      if ρc[i,j,k] < ρ_fl || isnan(ρc[i,j,k])
         ρ[i,j,k] = ρ_fl
        ρc[i,j,k] = ρ_fl
      end
      if p[i,j,k] < p_fl || isnan(p[i,j,k])
        e[i,j,k] = p_fl/γm1 + KE[i,j,k]
        p[i,j,k] = p_fl
      end
    end
  end
  return nothing
end



end