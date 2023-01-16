#module FluxIntegrator
include("ComputeFlux.jl");

function FluxIntegrator!(U::CuArray{T,4},dt::t,prob;Order=1) where {T,t}
    
    # Half time step Δt/2 for F^{n+1/2}
    # Every New Time Step
    # U (copy) -> U_half
    #  Update U_half from U + dFdx
    ComputeFlux!(prob;Order=Order);
    
    prob.usr_func(U, dt, prob);
    
    @timeit_debug prob.debugTimer "Adding ∂F∂x" CUDA.@sync begin
      Add∂F∂x!(U,dt,prob);
    end
    return nothing;
end

function Add∂F∂x_old!(u_out,weight,prob)
  # point to grid;
  grid = prob.grid;
  
  # get the information of the dAᵢ and dV  
  dA₁ = grid.dA₁;
  dA₂ = grid.dA₂;
  dA₃ = grid.dA₃;
  dV  =  grid.dV;
  dV⁻¹ = 1/dV;
  
  is,ie = grid.x1.is::Int,grid.x1.ie::Int;
  js,je = grid.x2.js::Int,grid.x2.je::Int;
  ks,ke = grid.x3.ks::Int,grid.x3.ke::Int;
  
  ∂ᵢ(A) = diff(A,dims=1)
  ∂ⱼ(A) = diff(A,dims=2)
  ∂ₖ(A) = diff(A,dims=3)
  
  # get the F,H,G Flux
  F,H,G = prob.flux.F,prob.flux.H,prob.flux.G
  #Compute dF/dx ≈ (F(i+1)-F(i))/dx 
  # X₁ direction
  if (grid.nx::Int >1) 
    Fᵢ = view(F,is:ie+1,js:je,ks:ke,:)
    @views u_out[is:ie,js:je,ks:ke,:] .-= weight/dV*dA₁*∂ᵢ(Fᵢ);
  end

  # X₂ direction
  if (grid.ny::Int >1)
    Fⱼ = view(G,is:ie,js:je+1,ks:ke,:)
    @views u_out[is:ie,js:je,ks:ke,:] .-= weight/dV*dA₂*∂ⱼ(Fⱼ);
  end

  # X₃ direction
  if (grid.nz::Int >1)
    Fₖ = view(H, is:ie,js:je,ks:ke+1,:)
    @views u_out[is:ie,js:je,ks:ke,:] .-= weight/dV*dA₃*∂ₖ(Fₖ);
  end

  return nothing;
end


function Add∂F∂x!(u_out,weight,prob)
  # point to grid;
  grid = prob.grid;
  
  # get the information of the dAᵢ and dV  
  dA₁  = grid.dA₁;
  dA₂  = grid.dA₂;
  dA₃  = grid.dA₃;
  dV   =  grid.dV;
  dV⁻¹ = 1/dV;
  T    = eltype(grid)

  nx,ny,nz = grid.nx::Int, grid.ny::Int, grid.nz::Int

  is,ie = grid.x1.is::Int,grid.x1.ie::Int;
  js,je = grid.x2.js::Int,grid.x2.je::Int;
  ks,ke = grid.x3.ks::Int,grid.x3.ke::Int;
  
  # get the F,H,G Flux
  F,H,G = prob.flux.F,prob.flux.H,prob.flux.G

  # Compute the Coef and convert the datatype T
  Coefx = T(weight/dV*dA₁)
  Coefy = T(weight/dV*dA₂)
  Coefz = T(weight/dV*dA₃)  

  #Compute dF/dx ≈ (F(i+1)-F(i))/dx 
  Compute∂F∂x!(u_out, F, G, H, 
                  nx, ny, nz,
               Coefx, Coefy, Coefz,
               is,ie,js,je,ks,ke)


  return nothing;
end


function Compute∂F∂x!(U::CuArray{T,4}, F::CuArray{T,4}, G::CuArray{T,4}, H::CuArray{T,4},
                      nx::Int, ny::Int, nz::Int, 
                      Coefx, Coefy, Coefz,
                      is, ie, js, je, ks, ke) where T
  
  Nhydro = size(F,4)
  threads = (32,8,1)
  blocks  = (ceil(Int, size(F,1)/threads[1]),ceil(Int, size(F,2)/threads[2]), ceil(Int,size(F,3)/threads[3]))
  for n = 1:Nhydro
    Ui = (@view  U[:,:,:,n])::CuArray{T,3} 
    Fi = (@view  F[:,:,:,n])::CuArray{T,3} 
    Gi = (@view  G[:,:,:,n])::CuArray{T,3} 
    Hi = (@view  H[:,:,:,n])::CuArray{T,3} 
    @cuda blocks = blocks threads = threads ∂F∂x_CUDA!(Ui, Fi, Gi, Hi,
                                                       nx, ny, nz,
                                                       Coefx, Coefy, Coefz,
                                                       is,ie,js,je,ks,ke)
  end
end


function ∂F∂x_CUDA!(U, F, G, H,
                    nx, ny, nz,
                    Coefx, Coefy, Coefz,
                    is,ie,js,je,ks,ke)
  #define the zero type
  T = eltype(U)
  ZERO = T(0.0)

  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is:ie+1) && nx > 1
    @inbounds ∂F∂x = Coefx*(F[i+1,j,k] - F[i,j,k])
  else
    ∂F∂x = ZERO
  end

  if k ∈ (ks:ke) && j ∈ (js:je+1) && i ∈ (is:ie) && ny > 1
    @inbounds ∂G∂y = Coefy*(G[i,j+1,k] - G[i,j,k])
  else
    ∂G∂y = ZERO
  end

  if k ∈ (ks:ke+1) && j ∈ (js:je) && i ∈ (is:ie) && nz > 1
    @inbounds ∂H∂z = Coefz*(H[i,j,k+1] - H[i,j,k])
  else
    ∂H∂z = ZERO
  end

  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is:ie)
    @inbounds U[i,j,k] -= (∂F∂x + ∂G∂y + ∂H∂z)
  end

  return nothing
end

#end
