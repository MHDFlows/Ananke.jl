#Module : Adiabatic HD Solver
#Date   : 17 Dec 2022
#Author : Ka Wai HO
#inspiration : Athena++ HLLC Solver 

@inline function HLLC!(fl, wl, wr, ivx::Int, 
                       is::Int, ie::Int, js::Int, je::Int, ks::Int, ke::Int, 
                       EOS, grid, tmp)
  # Declare the data type and index
  Nhydro  = grid.Nhydro::Int;
  IDN,IEN,IPR  = grid.ind.ρ::Int ,grid.ind.e::Int ,grid.ind.P::Int ;
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int;
     
  # Declare the physics parameter
  γ     = EOS.γ;
  γm1⁻¹ = 1f0/(γ - 1);
  ivy = IVX + ((ivx-IVX)+1)%3;
  ivz = IVX + ((ivx-IVX)+2)%3;
    
  # Define 3D sub-array/3D array "pointer" as CUDA supports 3D array better
  @views FL_IDN, WL_IDN, WR_IDN = fl[:,:,:,IDN], wl[:,:,:,IDN], wr[:,:,:,IDN] 
  @views FL_IVX, WL_IVX, WR_IVX = fl[:,:,:,ivx], wl[:,:,:,ivx], wr[:,:,:,ivx]
  @views FL_IVY, WL_IVY, WR_IVY = fl[:,:,:,ivy], wl[:,:,:,ivy], wr[:,:,:,ivy]
  @views FL_IVZ, WL_IVZ, WR_IVZ = fl[:,:,:,ivz], wl[:,:,:,ivz], wr[:,:,:,ivz] 
  @views FL_IEN, WL_IPR, WR_IPR = fl[:,:,:,IEN], wl[:,:,:,IPR], wr[:,:,:,IPR]

  # actual computation
  # 16*16 is the current solution as registers source is limited
  # FUTURE WORK : optimization
  threads = (32,16,1) # (8,8,8)
  blocks  = (ceil(Int,size(fl,1)/threads[1]),ceil(Int,size(fl,2)/threads[2]), ceil(Int,size(fl,3)/threads[3]))
  @cuda blocks = blocks threads = threads HLLC_CUDA!(FL_IDN, FL_IEN, FL_IVX, FL_IVY, FL_IVZ,
                                                     WL_IDN, WL_IPR, WL_IVX, WL_IVY, WL_IVZ,
                                                     WR_IDN, WR_IPR, WR_IVX, WR_IVY, WR_IVZ,
                                                     is, ie, js, je, ks, ke, γm1⁻¹, γ)
end

#lower level CUDA kernel function 
function HLLC_CUDA!(FL_IDN, FL_IEN, FL_IVX, FL_IVY, FL_IVZ,
                    WL_IDN, WL_IPR, WL_IVX, WL_IVY, WL_IVZ,
                    WR_IDN, WR_IPR, WR_IVX, WR_IVY, WR_IVZ,
                    is, ie, js, je, ks, ke, γm1⁻¹, γ)
  TINY_NUMBER = 1f-18
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is:ie)
    @inbounds WLI_IDN = WL_IDN[i,j,k]
    @inbounds WLI_IPR = WL_IPR[i,j,k]
    @inbounds WLI_IVX = WL_IVX[i,j,k]
    @inbounds WLI_IVY = WL_IVY[i,j,k]
    @inbounds WLI_IVZ = WL_IVZ[i,j,k]

    @inbounds WRI_IDN = WR_IDN[i,j,k]
    @inbounds WRI_IPR = WR_IPR[i,j,k]
    @inbounds WRI_IVX = WR_IVX[i,j,k]
    @inbounds WRI_IVY = WR_IVY[i,j,k]
    @inbounds WRI_IVZ = WR_IVZ[i,j,k]
        
    # Compute the actual flux
    cl = √(γ*WLI_IPR/WLI_IDN)
    cr = √(γ*WRI_IPR/WRI_IDN)
    el = WLI_IPR*γm1⁻¹ + 0.5*WLI_IDN*(WLI_IVX*WLI_IVX + WLI_IVY*WLI_IVY + WLI_IVZ*WLI_IVZ)
    er = WRI_IPR*γm1⁻¹ + 0.5*WRI_IDN*(WRI_IVX*WRI_IVX + WRI_IVY*WRI_IVY + WRI_IVZ*WRI_IVZ)

    ρa = 0.5*( WLI_IDN + WRI_IDN )
    ca = 0.5*( cl + cr )
    pmid = 0.5*(WLI_IPR + WRI_IPR + (WLI_IVX-WRI_IVX)*ρa*ca)
    umid = 0.5*(WLI_IVX + WRI_IVX + (WLI_IPR-WRI_IPR)/ρa/ca)
    ρL  = WLI_IDN + (WLI_IVX - umid)*ρa/ca 
    ρR  = WRI_IDN + (umid - WRI_IVX)*ρa/ca

    ql = pmid <= WLI_IPR ? 1.0 : 
                √(1.0 + (γ + 1)/(2γ)*(pmid/WLI_IPR - 1.0))
    qr = pmid <= WRI_IPR ? 1.0 : 
                √(1.0 + (γ + 1)/(2γ)*(pmid/WRI_IPR - 1.0))

    al = WLI_IVX - cl*ql
    ar = WRI_IVX + cr*qr

    bp = ar > 0.0 ? ar :  TINY_NUMBER
    bm = al < 0.0 ? al : -TINY_NUMBER

    vxl = WLI_IVX - al
    vxr = WRI_IVX - ar

    tl = WLI_IPR + vxl*WLI_IDN*WLI_IVX
    tr = WRI_IPR + vxr*WRI_IDN*WRI_IVX    

    ml =  WLI_IDN*vxl
    mr = -WRI_IDN*vxr

    am = (tl - tr) / (ml + mr)
    cp = (ml*tr + mr*tl)/(ml + mr)
    cp = cp > 0 ? cp : 0.0

    vxl = WLI_IVX - bm
    vxr = WRI_IVX - bp

    FLI_DN = WLI_IDN*vxl
    FRI_DN = WRI_IDN*vxr

    FLI_VX = WLI_IDN*WLI_IVX*vxl + WLI_IPR
    FRI_VX = WRI_IDN*WRI_IVX*vxr + WRI_IPR

    FLI_VY = WLI_IDN*WLI_IVY*vxl
    FRI_VY = WRI_IDN*WRI_IVY*vxr

    FLI_VZ = WLI_IDN*WLI_IVZ*vxl
    FRI_VZ = WRI_IDN*WRI_IVZ*vxr

    FLI_EN = el*vxl + WLI_IPR*WLI_IVX
    FRI_EN = er*vxr + WRI_IPR*WRI_IVX

    if am >= 0.0
      sl =  am/(am-bm)
      sr =  0.0
      sm = -bm/(am-bm)
    else
      sl =  0.0
      sr = -am/(bp-am)
      sm =  bp/(bp-am)
    end
 
    # Move the result to the output data array
    @inbounds FL_IDN[i,j,k] = sl*FLI_DN + sr*FRI_DN
    @inbounds FL_IVX[i,j,k] = sl*FLI_VX + sr*FRI_VX + sm*cp
    @inbounds FL_IVY[i,j,k] = sl*FLI_VY + sr*FRI_VY
    @inbounds FL_IVZ[i,j,k] = sl*FLI_VZ + sr*FRI_VZ
    @inbounds FL_IEN[i,j,k] = sl*FLI_EN + sr*FRI_EN + sm*cp*am

  end 
  return nothing
end