using MPI
include("ZCurve/ZOrder.jl")
include("../Datastructure.jl")
include("../bval/BoundaryExchange.jl")
include("../bval/outflow.jl")
include("../bval/periodic.jl")
include("../bval/reflective.jl")

function Get_MPI_Gobal_Index(nx::Int,ny::Int,nz::Int,LL::Array,Rank)
  i,j,k = Tuple(findall(LL.==Rank)[1])
  i0,j0,k0 = i-1, j-1, k-1
	dN = [nx,ny,nz]
	x1s, x2s, x3s = ([i0,j0,k0].*dN).+1
	x1e, x2e, x3e = ([i0,j0,k0].+1).*dN
	return x1s,x1e,x2s,x2e,x3s,x3e
end

function Get_Neighbor(LL0,Rank)
  # Looking for the neighbor
  i,j,k = Tuple(findall(LL0.==Rank)[1])
  i,j,k = i+1,j+1,k+1
  LLN = GetLLwithNeighbor(LL0)

  XL_Neighbor = LLN[i-1,j  ,k  ]
  XR_Neighbor = LLN[i+1,j  ,k  ] 
  YL_Neighbor = LLN[i  ,j-1,k  ]
  YR_Neighbor = LLN[i  ,j+1,k  ]
  ZL_Neighbor = LLN[i  ,j  ,k-1]
  ZR_Neighbor = LLN[i  ,j  ,k+1]
  #println("I am rank :$(Rank)")
  #size(LL0)[1] > 0 ? println("XL_Neighbor = $(XL_Neighbor), XR_Neighbor = $(XR_Neighbor)") : nothing
  #size(LL0)[2] > 0 ? println("YL_Neighbor = $(YL_Neighbor), YR_Neighbor = $(YR_Neighbor)") : nothing
  #size(LL0)[3] > 0 ? println("ZL_Neighbor = $(ZL_Neighbor), ZR_Neighbor = $(ZR_Neighbor)") : nothing

  return  [XL_Neighbor, XR_Neighbor,
           YL_Neighbor, YR_Neighbor,
           ZL_Neighbor, ZR_Neighbor]
end

function GetLLwithNeighbor(LL)
  nx,ny,nz = size(LL)
  LLN = zeros(nx+2,ny+2,nz+2)
  @views LLN[2:end-1,2:end-1,2:end-1] .= LL
  @views LLN[1      ,2:end-1,2:end-1] .= LL[end,:,:]
  @views LLN[2:end-1,      1,2:end-1] .= LL[:,end,:]
  @views LLN[2:end-1,2:end-1,      1] .= LL[:,:,end]
  @views LLN[    end,2:end-1,2:end-1] .= LL[1,:,:]
  @views LLN[2:end-1,    end,2:end-1] .= LL[:,1,:]
  @views LLN[2:end-1,2:end-1,    end] .= LL[:,:,1]
  return LLN
end

function Boundary_Function(x1s,x1e,x2s,x2e,x3s,x3e,
						               Ncx, Ncy, Ncz, Boundary)

  Bval_func_list = []
  for (xis,xie,N,BoundaryL,BoundaryR,i) in zip((x1s,x2s,x3s),(x1e,x2e,x3e),(Ncx,Ncy,Ncz),
                                                Boundary[[1,3,5]],Boundary[[2,4,6]],(1,2,3))
    if N > 1
      # meta programming for pointing the function
      #example : X1L_Outflow_MPI!/X1L_Outflow!
      X???L_BoundaryExchange! = eval(Symbol(:X,i,:L_,BoundaryL,:_MPI!))
      X???R_BoundaryExchange! = eval(Symbol(:X,i,:R_,BoundaryL,:_MPI!))
    else
      X???L_BoundaryExchange! = eval(Symbol(:X,i,:L_,BoundaryL,:!))
      X???R_BoundaryExchange! = eval(Symbol(:X,i,:R_,BoundaryL,:!))
    end
    push!(Bval_func_list,X???L_BoundaryExchange!)
    push!(Bval_func_list,X???R_BoundaryExchange!)
  end

	return Bval_func_list[:]
end

function MPIGridConstruction(Nx,Ny,Nz, nx, ny, nz, Lx,Ly,Lz 
    	             				   T = Float64, Bfield = false, Nghost = 2, 
			             			     Boundary = ["Outflow","Outflow",
			             			                 "Outflow","Outflow",
			             			                 "Outflow","Outflow"],test=false)
    # Be Careful :
    # MPI.ie != grid.x1.x1e as MPI points to global index while x1 is for local index

    # U -> ??,e,p1,p2,p3
    # W -> ??,p,v1,v2,v3
    if Bfield == true
        error("Under Construction!\n")
    else
        ind = Ind_struct(1,2,2,3,4,5,3,4,5)
        Nhydro = 5
        NWave  = Nhydro 
    end
    
    if size(Lx)!=size(Ly)!=size(Lz)!=2
        error("Size of Length Array should be 2!\n") 
    end
    
    # Define the MPI parameters
    MPI.Init()
    comm = MPI.COMM_WORLD
    Comm_rank = MPI.Comm_rank
    #Get the total pros called by the MPI
    nprocs    = MPI.Comm_size(comm)
    # Get total core number defined by the user
    Ncx,Ncy,Ncz = div(Nx,nx),div(Ny,ny),div(Nz,nz)
    if nprocs != Ncx*Ncy*Ncz
      error("MPI pros: ($(nprocs) pros) and user defined pros: ($(Ncx*Ncy*Ncz) procs) is not matched ")
    end
    LL = GetRankMap(Ncx,Ncy,Ncz).-1
    Neighbor_ind = Get_Neighbor(LL,Comm_rank(comm))
    x1s,x1e,x2s,x2e,x3s,x3e = Get_MPI_Gobal_Index(nx,ny,nz,LL,Comm_rank(comm))
    MPI_info = MPI_Info(x1s,x1e,x2s,x2e,x3s,x3e,LL,
                        Neighbor_ind[1],Neighbor_ind[2],
                        Neighbor_ind[3],Neighbor_ind[4],
                        Neighbor_ind[5],Neighbor_ind[6],
                        MPI)
    
    # Define Local Length information 
    dLx  ,dLy  ,dLz   = diff(Lx)[1], diff(Ly)[1], diff(Lz)[1]
    Lx_st,Ly_st,Lz_st = (x1s-1)*dLx/Nx,(x2s-1)*dLy/Ny,(x3s-1)*dLz/Nz
    Lx_ed,Ly_ed,Lz_ed = (x1e-0)*dLx/Nx,(x2e-0)*dLy/Ny,(x3e-0)*dLz/Nz

    # Get the face/edge/Area information from all the dimenion
    x1f,x1v,??x1f,??x1v = GetFace_n_Edge(Lx_st,Lx_ed,nx,Nghost)
    x2f,x2v,??x2f,??x2v = GetFace_n_Edge(Ly_st,Ly_ed,ny,Nghost)
    x3f,x3v,??x3f,??x3v = GetFace_n_Edge(Lz_st,Lz_ed,nz,Nghost)

    dA??? = ??x2f[1]*??x3f[1]
    dA??? = ??x1f[1]*??x3f[1]
    dA??? = ??x1f[1]*??x2f[1]
    dV  = ??x1f[1]*??x2f[1]*??x3f[1]

    # Check if user define the Boundary correctly
    N_B = length(findall(Boundary.=="outflow")) +length(findall(Boundary.=="periodic")) +
          length(findall(Boundary.=="reflective"))
    N_B == 6 ? nothing : error("Boundary is not declared correctly!")

    X???L_BE!, X???R_BE!, X???L_BE!, X???R_BE!, X???L_BE!, X???R_BE! = Boundary_Function(x1s,x1e,x2s,x2e,x3s,x3e,
	  					    											          	                         Ncx,Ncy,Ncz, Boundary)
    
    # Construct the local index and it if it is 1/2D case
    is,ie = Nx == 1 ? (1,1) : (1+Nghost, Nx+Nghost)
    js,je = Ny == 1 ? (1,1) : (1+Nghost, Ny+Nghost)
    ks,ke = Nz == 1 ? (1,1) : (1+Nghost, Nz+Nghost)

    Nx_tot = Nx == 1 ?  1 : nx + 2*Nghost  
    Ny_tot = Ny == 1 ?  1 : ny + 2*Nghost
    Nz_tot = Nz == 1 ?  1 : nz + 2*Nghost

    BvalA = CUDA.ones(T,(Nghost, Ny_tot,Nz_tot, Nhydro))  
    BvalB = CUDA.ones(T,(Nx_tot, Nghost,Nz_tot, Nhydro))
    BvalC = CUDA.ones(T,(Nx_tot, Ny_tot,Nghost, Nhydro))

    x1  = x???_struct(is, ie, x1f, x1v, ??x1f, ??x1v, BvalA,
                    X???L_BE!,X???R_BE!)
    x2  = x???_struct(js, je, x2f, x2v, ??x2f, ??x2v, BvalB,
                    X???L_BE!,X???R_BE!)
    x3  = x???_struct(ks, ke, x3f, x3v, ??x3f, ??x3v, BvalC,
                    X???L_BE!,X???R_BE!)

    return  MPI_Grid( Nx,Ny,Nz,nx,ny,nz,Nghost,
                      Lx_st,Ly_st,Lz_st,
                      Lx_ed,Ly_ed,Lz_ed,
                      dA???,dA???,dA???,dV,
                      x1,x2,x3,ind,Nhydro,NWave,MPI_info)
end
