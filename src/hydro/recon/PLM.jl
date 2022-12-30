#Module : 2nd order reconstruction function
#Date   : 17 Dec 2022
#Author : Ka Wai HO

# dispath function
PiecewiseLinearX₁!(w, wl, wr, grid::AbstractGrid) = PiecewiseLinearXᵢ!(w, wl, wr, grid; dir="1")
PiecewiseLinearX₂!(w, wl, wr, grid::AbstractGrid) = PiecewiseLinearXᵢ!(w, wl, wr, grid; dir="2")
PiecewiseLinearX₃!(w, wl, wr, grid::AbstractGrid) = PiecewiseLinearXᵢ!(w, wl, wr, grid; dir="3")

function PiecewiseLinearXᵢ!(w, wl, wr,
                            grid::AbstractGrid; dir="x")

  T = eltype(grid)
  # Get back the infromation of all dimonions
  is,ie = grid.x1.is::Int, grid.x1.ie::Int;
  js,je = grid.x2.js::Int, grid.x2.je::Int;
  ks,ke = grid.x3.ks::Int, grid.x3.ke::Int;

  Nhydro = grid.Nhydro::Int
  # generating x1/x2/x3
    xi = getproperty(grid,Symbol("x",dir))
   xif = getproperty(  xi,Symbol("x",dir,"f"))
   xiv = getproperty(  xi,Symbol("x",dir,"v"))
  Δxif = getproperty(  xi,Symbol("Δx",dir,"f"))
  
  # CUDA threads & PLM function setup
  # Tx*Ty*Tz = 1024
  # Tx = T0*Nᵢ/∑ᵢ(Nᵢ-1)
  threads = ( 32, 8, 1) #(9,9,9)
  blocks   = ( ceil(Int,size(w,1)/threads[1]), ceil(Int,size(w,2)/threads[2]), ceil(Int,size(w,3)/threads[3]))  

  if dir=="1"
    PLMXᵢ_CUDA! = PLMX₁_CUDA!
  elseif dir == "2"
    PLMXᵢ_CUDA! = PLMX₂_CUDA!
  elseif dir == "3"
    PLMXᵢ_CUDA! = PLMX₃_CUDA!
  end
  for n = 1:Nhydro
     w_3D = (@view  w[:,:,:,n])::CuArray{T,3} 
    wl_3D = (@view wl[:,:,:,n])::CuArray{T,3}
    wr_3D = (@view wr[:,:,:,n])::CuArray{T,3}
    @cuda blocks = blocks threads = threads PLMXᵢ_CUDA!(w_3D, wl_3D, wr_3D,
                                                        is, ie, js, je, ks, ke,
                                                        xif, xiv, Δxif)
  end
  return nothing
end

function PLMX₁_CUDA!(w, wl, wr,
                     is, ie, js, je, ks, ke,
                     x1f, x1v, Δx1f)
    
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is-1:ie+1)
      @inbounds Δwl = (w[i  ,j,k] - w[i-1,j,k])
      @inbounds Δwr = (w[i+1,j,k] - w[i  ,j,k])
    
      Δw² = Δwl*Δwr
      Δwm = Δw² <= 0.0 ? 0.0 : 2.0*Δw²/(Δwl+Δwr)
    
      @inbounds wl[i+1,j,k] = w[i,j,k] + (x1f[i+1] - x1v[i])/Δx1f[i]*Δwm
      @inbounds wr[i  ,j,k] = w[i,j,k] - (x1v[i  ] - x1f[i])/Δx1f[i]*Δwm
  end
    return nothing
end

function PLMX₂_CUDA!(w,  wl, wr,
                     is, ie, js, je, ks, ke,
                     x2f, x2v, Δx2f)
    
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  if k ∈ (ks:ke) && i ∈ (is:ie) && j ∈ (js-1:je+1)  
      @inbounds Δwl = (w[i,j  ,k] - w[i,j-1,k])
      @inbounds Δwr = (w[i,j+1,k] - w[i,j  ,k])
      
      Δw² = Δwl*Δwr
      Δwm = Δw² <= 0.0 ? 0.0 : 2.0*Δw²/(Δwl+Δwr)
    
      @inbounds wl[i,j+1,k] = w[i,j,k] + (x2f[j+1] - x2v[j])/Δx2f[j]*Δwm
      @inbounds wr[i,j  ,k] = w[i,j,k] - (x2v[j  ] - x2f[j])/Δx2f[j]*Δwm
  end
  return nothing
end

function PLMX₃_CUDA!(w,  wl, wr,
                     is, ie, js, je, ks, ke,
                     x3f, x3v, Δx3f)
    
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 

  if j ∈ (js:je) && i ∈ (is:ie) && k ∈ (ks-1:ke+1)
      @inbounds Δwl = (w[i,j,k  ] - w[i,j,k-1])
      @inbounds Δwr = (w[i,j,k+1] - w[i,j,k  ])
    
      Δw² = Δwl*Δwr
      Δwm = Δw² <= 0.0 ? 0.0 : 2.0*Δw²/(Δwl+Δwr)
    
      @inbounds wl[i,j,k+1] = w[i,j,k] + (x3f[k+1] - x3v[k])/Δx3f[k]*Δwm
      @inbounds wr[i,j,k  ] = w[i,j,k] - (x3v[k  ] - x3f[k])/Δx3f[k]*Δwm
    
  end
  return nothing
end
