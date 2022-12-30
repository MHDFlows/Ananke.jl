# Isothermal HD Solver (HLLE)
# Date : 27 Dec 2022 
# Author : Ka Wai HO
# Comment : Modified from athena++ HLLE module, we skipped adabatic gas implemention

function HLLE!(fl, wl, wr, ivx::Int, 
               is::Int, ie::Int, js::Int, je::Int, ks::Int, ke::Int, 
               EOS, grid, tmp)
  # Declare the data type and index
  Nhydro  = grid.Nhydro::Int
  IDN     = grid.ind.ρ::Int
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int
     
  # Declare the physics parameter
  cₛ   = EOS.cₛ
  ivy = IVX + ((ivx-IVX)+1)%3
  ivz = IVX + ((ivx-IVX)+2)%3

  # Define 3D sub-array/3D array "pointer" as CUDA supports 3D array only
  @views FL_IDN, WL_IDN, WR_IDN = fl[:,:,:,IDN], wl[:,:,:,IDN], wr[:,:,:,IDN] 
  @views FL_IVX, WL_IVX, WR_IVX = fl[:,:,:,ivx], wl[:,:,:,ivx], wr[:,:,:,ivx]
  @views FL_IVY, WL_IVY, WR_IVY = fl[:,:,:,ivy], wl[:,:,:,ivy], wr[:,:,:,ivy]
  @views FL_IVZ, WL_IVZ, WR_IVZ = fl[:,:,:,ivz], wl[:,:,:,ivz], wr[:,:,:,ivz] 

  # actual computation
  # 16*16 is the current solution as registers source is limited
  # FUTURE WORK : optimization
  threads = (8,8,8)
  blocks  = (ceil(Int,size(fl,1)/threads[1]),ceil(Int,size(fl,2)/threads[2]), ceil(Int,size(fl,3)/threads[3]))
  @cuda blocks = blocks threads = threads HLLE_CUDA!(FL_IDN, FL_IVX, FL_IVY, FL_IVZ,
                                                     WL_IDN, WL_IVX, WL_IVY, WL_IVZ,
                                                     WR_IDN, WR_IVX, WR_IVY, WR_IVZ,
                                                     is, ie, js, je, ks, ke, cₛ)
end

function HLLE_CDUA!(FL_IDN, FL_IVX, FL_IVY, FL_IVZ,
                    WL_IDN, WL_IVX, WL_IVY, WL_IVZ,
                    WR_IDN, WR_IVX, WR_IVY, WR_IVZ,
                    is, ie, js, je, ks, ke, cₛ)
  #define the i,j,k
  i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
  j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
  k = (blockIdx().z - 1) * blockDim().z + threadIdx().z 
  if k ∈ (ks:ke) && j ∈ (js:je) && i ∈ (is:ie)
    # Step 1. Read the L/R vars from the main VRAM
    wli_IDN = WL_IDN[i,j,k]
    wli_IVX = WL_IVX[i,j,k]
    wli_IVY = WL_IVY[i,j,k]
    wli_IVZ = WL_IVZ[i,j,k]

    wri_IDN = WR_IDN[i,j,k]
    wri_IVX = WR_IVX[i,j,k]
    wri_IVY = WR_IVY[i,j,k]
    wri_IVZ = WR_IVZ[i,j,k]

    sqrtdl = √(wli_IDN)
    sqrtdr = √(wri_IDN)
    isdlpdr = 1.0/(sqrtdl + sqrtdr)

    wroe_IDN = sqrtdl*sqrtdr
    wroe_IVX = (sqrtdl*wli_IVX + sqrtdr*wri_IVX)*isdlpdr
    wroe_IVY = (sqrtdl*wli_IVY + sqrtdr*wri_IVY)*isdlpdr
    wroe_IVZ = (sqrtdl*wli_IVZ + sqrtdr*wri_IVZ)*isdlpdr

    # Step 3.  Compute sound speed in L,R, and Roe-averaged states
    cl = cr = a = cₛ
    
    # Step 4. Compute the max/min wave speeds based on L/R and Roe-averaged values
    al = min((wroe_IVX - a),(wli_IVX - cl))
    ar = max((wroe_IVX + a),(wri_IVX + cr))

    bp = ar > 0.0 ? ar : 0.0
    bm = al < 0.0 ? al : 0.0

    #  Step 5. Compute L/R fluxes along lines bm/bp: F_L - (S_L)U_L F_R - (S_R)U_R
    vxl = wli_IVX - bm
    vxr = wri_IVX - bp

    fl_IDN = wli_IDN*vxl
    fr_IDN = wri_IDN*vxr

    fl_IVX = wli_IDN*wli_IVX*vxl+(cₛ*cₛ)*wli_IDN
    fr_IVX = wri_IDN*wri_IVX*vxr+(cₛ*cₛ)*wri_IDN

    fl_IVY = wli_IDN*wli_IVY*vxl
    fr_IVY = wri_IDN*wri_IVY*vxr

    fl_IVZ = wli_IDN*wli_IVZ*vxl
    fr_IVZ = wri_IDN*wri_IVZ*vxr

    # Step 6. Compute the HLLE flux at interface.
    tmp = bp == bm ? 0.0 : 0.5*(bp + bm)/(bp - bm)

    FL_IDN[i,j,k] = 0.5*(fl_IDN+fr_IDN) + (fl_IDN - fr_IDN)*tmp
    FL_IVX[i,j,k] = 0.5*(fl_IVX+fr_IVX) + (fl_IVX - fr_IVX)*tmp
    FL_IVY[i,j,k] = 0.5*(fl_IVY+fr_IVY) + (fl_IVY - fr_IVY)*tmp
    FL_IVZ[i,j,k] = 0.5*(fl_IVZ+fr_IVZ) + (fl_IVZ - fr_IVZ)*tmp

  end
  return nothing
end