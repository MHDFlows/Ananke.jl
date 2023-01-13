"Module For Problem Construction"

include("structure/data_structure.jl");
include("structure/grid/grid.jl")
#include("grid/MPI_grid.jl")
include("bval/BoundaryExchange.jl")
include("bval/outflow.jl")
include("bval/periodic.jl")
include("bval/reflective.jl")
include("hydro/recon/PLM.jl");
include("hydro/recon/PPM.jl");
include("hydro/recon/DC.jl");
include("hydro/Solver/HLLC.jl");
include("hydro/Solver/HLLE.jl");
include("eos/Adiabatic.jl");
include("eos/Isothermal.jl");

export ProblemConstruction

nothingfunction(args...) = nothing;
#Construct a Problem
function ProblemConstrcut( ; N = (64,1,1),
                             n = (64,1,1),
                            Lx = (0.0,1.0),
                            Ly = Lx,
                            Lz = Lx,
                  SpatialOrder = 2,  
                    TimelOrder = 2,
                            T  = Float32,
                        Bfield = false,
                        Nghost = 2,
                       EOSType = "Adiabatic",
                      Boundary = ["Outflow","Outflow",
                                  "Outflow","Outflow",
                                  "Outflow","Outflow"],
                         Coord = "Cartesian",
                       usr_foo = nothingfunction)
   
    Nx,Ny,Nz = N;
    nx,ny,nz = n;
    if Nx!=nx || Ny!=ny || Nz!=nz
      grid = MPIGridConstruction(Nx,Ny,Nz, nx, ny, nz, Lx,Ly,Lz; 
                                 T = T, Bfield = Bfield, EOSType = EOSType,
                                 Nghost = Nghost, Boundary = Boundary)
    else
      # define the grid
      grid = GridConstruction(Nx,Ny,Nz, nx, ny, nz, Lx,Ly,Lz; 
                              T = T, Bfield = Bfield, EOSType = EOSType,
                              Nghost = Nghost, Boundary=Boundary) 
    end

    # define the EOS
    EOS  = EOSConstruction(EOSType)
    # define the flux
    flux = FluxConstruction(grid)
    # defin the sol
    sol  = SolConstruction(flux)
    # dfeine the clock
    clock = Clock(0.0,0.0,1)
    # define the equation
    Eq    = EquationConstruction(grid, SpatialOrder,TimelOrder; 
                                 Bfield=Bfield, EOSType=EOSType)
    # define the sketch array for solver
    Solver = tmpConstruction(grid)

    # define the flag
    Flag = FlagConstruction(Bfield,Coord,SpatialOrder,TimelOrder,Boundary)


    # define the debug timer
    to = TimerOutput();

   return Problem(EOS, grid, flux, sol, clock,
                   Eq, usr_foo, Solver, Flag, to);

end

function FlagConstruction(B,Coord,SpatialOrder,TimelOrder,Boundary)
  return Flag(B, Coord, SpatialOrder, TimelOrder, Boundary[:]...)
end

function SolConstruction(flux)

    U = copy(flux.F);
    W = copy(flux.F);
    U_half = copy(flux.F);
    
    return Sol(U,W,U_half)
end

function EquationConstruction(grid,SpatialOrder,TimeOrder;Bfield=false,EOSType="Adiabatic")
    nothingfunction(args...) = nothing;
    if SpatialOrder == TimeOrder == 2
      ReconstructionFunction_X₁! = PiecewiseLinearX₁!;
      ReconstructionFunction_X₂! = ifelse(grid.Ny > 1 , PiecewiseLinearX₂!,nothingfunction);
      ReconstructionFunction_X₃! = ifelse(grid.Nz > 1 , PiecewiseLinearX₃!,nothingfunction);
    elseif SpatialOrder == 1;
      ReconstructionFunction_X₁! = DonorCellX₁!;
      ReconstructionFunction_X₂! = ifelse(grid.Ny > 1 , DonorCellX₂!,nothingfunction);
      ReconstructionFunction_X₃! = ifelse(grid.Nz > 1 , DonorCellX₃!,nothingfunction);    
    elseif SpatialOrder == 3
      ReconstructionFunction_X₁! = PiecewiseParabolicX₁!
      ReconstructionFunction_X₂! = ifelse(grid.Ny > 1 , PiecewiseParabolicX₂!,nothingfunction);
      ReconstructionFunction_X₃! = ifelse(grid.Nz > 1 , PiecewiseParabolicX₃!,nothingfunction);
    else
        error("Sorry, We only support spatail/time order = 1/2/3 : (");
    end
    
    if Bfield == true
      error("Under Construction");
    else
      RoeSolver! = EOSType == "Adiabatic" ? HLLC! : HLLE!
    end 

    BoundaryExchangeFunction! = BoundaryExchange!;
        
    return               Equation(DonorCellX₁!,
                                  DonorCellX₂!,
                                  DonorCellX₃!,
                    ReconstructionFunction_X₁!,
                    ReconstructionFunction_X₂!,
                    ReconstructionFunction_X₃!,
                                    RoeSolver!,
                                  SpatialOrder,
                                     TimeOrder,
                     BoundaryExchangeFunction!);
end

function FluxConstruction(grid)
    T        = eltype(grid)
    nx,ny,nz = grid.nx,grid.ny,grid.nz;
    Nghost   = grid.Nghost;
    Nhydro   = grid.Nhydro;

    # Correct the nx,ny,nz if it is 1/2D case
    nx_tot = nx == 1 ?  1 : nx + 2*Nghost  
    ny_tot = ny == 1 ?  1 : ny + 2*Nghost
    nz_tot = nz == 1 ?  1 : nz + 2*Nghost

    F = CUDA.zeros(T,(nx_tot, ny_tot, nz_tot, Nhydro) ) 
    G = ifelse(ny > 1, copy(F), CUDA.zeros(T,1,1,1,Nhydro));
    H = ifelse(nz > 1, copy(F), CUDA.zeros(T,1,1,1,Nhydro));
    return Flux(F,G,H);
end

function GridConstruction(Nx,Ny,Nz,nx,ny,nz, Lx,Ly,Lz;
                          T = Float32, Bfield = false, EOSType = "Adiabatic",
                          Nghost = 2, 
                          Boundary = ["outflow","outflow",
                                      "outflow","outflow",
                                      "outflow","outflow"])
  # U -> ρ,p1,p2,p3,,e (if Adiabatic)
  # W -> ρ,v1,v2,v3,,p (if Adiabatic)
  if Bfield == true
    error("Under Construction!\n");
  else
    # Slipt the ind_struct For the Future
    ind = Ind_struct(1,5,5,2,3,4,2,3,4);
    if EOSType == "Adiabatic"
      Nhydro = 5;
      NWave  = Nhydro; 
    elseif EOSType == "Isothermal"
      Nhydro = 4;
      NWave  = Nhydro; 
    end
  end
    
  if size(Lx)!=size(Ly)!=size(Lz)!=2
    error("Size of Length Array should be 2!\n"); 
  end
    
  # Define dimenion and grid structure 
  Lx_st,Ly_st,Lz_st = T(Lx[1]),T(Ly[1]),T(Lz[1])
  Lx_ed,Ly_ed,Lz_ed = T(Lx[2]),T(Ly[2]),T(Lz[2])

  # X₁ dimenion
  x1f,x1v,Δx1f,Δx1v = GetFace_n_Edge(Lx_st,Lx_ed,Nx,Nghost;T=T);

  # X₂ dimenion
  x2f,x2v,Δx2f,Δx2v = GetFace_n_Edge(Ly_st,Ly_ed,Ny,Nghost;T=T);

  # X₃ dimenion
  x3f,x3v,Δx3f,Δx3v = GetFace_n_Edge(Lz_st,Lz_ed,Nz,Nghost;T=T);

  dA₁ = CUDA.@allowscalar Δx2f[1]*Δx3f[1];
  dA₂ = CUDA.@allowscalar Δx1f[1]*Δx3f[1];
  dA₃ = CUDA.@allowscalar Δx1f[1]*Δx2f[1];
  dV  = CUDA.@allowscalar Δx1f[1]*Δx2f[1]*Δx3f[1];

  # Check if user define the Boundary correctly

  Bval_func_list = []
  for (BoundaryL,BoundaryR,i) in zip( Boundary[[1,3,5]],Boundary[[2,4,6]],(1,2,3))
    if eltype(BoundaryL) == Char 
      XᵢL_BoundaryExchange! = eval(Symbol(:X,i,:L_,BoundaryL,:!))
      push!(Bval_func_list,XᵢL_BoundaryExchange!)
    else
      push!(Bval_func_list,BoundaryL)
    end
    if eltype(BoundaryR) == Char
      XᵢR_BoundaryExchange! = eval(Symbol(:X,i,:R_,BoundaryL,:!))  
      push!(Bval_func_list,XᵢR_BoundaryExchange!)
    else
      push!(Bval_func_list,BoundaryR)
    end
  end

  # Correct the nx,ny,nz if it is 1/2D case
  is,ie = Nx == 1 ? (1,1) : (1+Nghost, Nx+Nghost)
  js,je = Ny == 1 ? (1,1) : (1+Nghost, Ny+Nghost)
  ks,ke = Nz == 1 ? (1,1) : (1+Nghost, Nz+Nghost)

  Nx_tot = Nx == 1 ?  1 : nx + 2*Nghost  
  Ny_tot = Ny == 1 ?  1 : ny + 2*Nghost
  Nz_tot = Nz == 1 ?  1 : nz + 2*Nghost

  BvalA = CUDA.ones(T,(Nghost, Ny_tot,Nz_tot, Nhydro))  
  BvalB = CUDA.ones(T,(Nx_tot, Nghost,Nz_tot, Nhydro))
  BvalC = CUDA.ones(T,(Nx_tot, Ny_tot,Nghost, Nhydro))

  x1  = x₁_struct(is, ie, x1f, x1v, Δx1f, Δx1v, BvalA,
                  Bval_func_list[1], Bval_func_list[2])
  x2  = x₂_struct(js, je, x2f, x2v, Δx2f, Δx2v, BvalB,
                  Bval_func_list[3], Bval_func_list[4])
  x3  = x₃_struct(ks, ke, x3f, x3v, Δx3f, Δx3v, BvalC,
                  Bval_func_list[5], Bval_func_list[6])

  return  Grid( Nx, Ny, Nz, Nx, Ny, Nz, Nghost,
                Lx_st, Ly_st, Lz_st,
                Lx_ed, Ly_ed, Lz_ed,
                dA₁, dA₂, dA₃, dV,
                x1, x2, x3, ind, Nhydro, NWave)
end

function GetFace_n_Edge(L_st,L_ed,N,Nghost;T=Float32)
  #Warning : For Nᵢ > 10000, Float32 data type will affect the calculation 
  # computation of determining the correct shape of x1f
  L_st ,L_ed = Float64(L_st), Float64(L_ed)
  Δx   = ifelse( N==1, (L_ed-L_st),(L_ed-L_st)/N);
  x1f  = collect(L_st:Δx:L_ed-Δx);
  G1f1 = collect(1:1:Nghost)*Δx  .+ L_st .- Nghost*Δx
  G1f2 = collect(1:1:Nghost)*Δx  .+ L_ed .-      0*Δx
  x1f  = vcat(G1f1,x1f,G1f2)
  x1v  = (x1f[2:end] + x1f[1:end-1])/2;
  Δx1f = diff(x1f); 
  Δx1v = copy(Δx1f); #diff(x1v) for future;
  return CuArray(T.(x1f)),CuArray(T.(x1v)),CuArray(T.(Δx1f)),CuArray(T.(Δx1v))
end

function EOSConstruction(EOSType)
  if EOSType == "Isothermal"
    γ = 1.0;
    PrimitivetoConserved! = Isothermal.PrimitivetoConserved!
    ConservedToPrimitive! = Isothermal.ConservedToPrimitive!
  elseif EOSType == "Adiabatic"
    γ = 5/3;
    PrimitivetoConserved! = Adiabatic.PrimitivetoConserved!
    ConservedToPrimitive! = Adiabatic.ConservedToPrimitive!
  else
    error("Unkown EOS : $EOSType !")
  end
 return EOS(EOSType, γ, 0.0,ConservedToPrimitive!,
                            PrimitivetoConserved!);
end

function tmpConstruction(grid)

  T = eltype(grid)
  nx,ny,nz  = grid.nx,grid.ny,grid.nz
  Nghost = grid.Nghost::Int
  Nhydro = grid.Nhydro::Int

  # Correct the nx,ny,nz if it is 1/2D case
  nx_tot = nx == 1 ?  1 : nx + 2*Nghost  
  ny_tot = ny == 1 ?  1 : ny + 2*Nghost
  nz_tot = nz == 1 ?  1 : nz + 2*Nghost
 
  wl  = CUDA.zeros(T,(nx_tot, ny_tot, nz_tot, Nhydro))
  wr  = CUDA.zeros(T,(nx_tot, ny_tot, nz_tot, Nhydro))
  
  return tmp_struct(wl,wr)

end

#Future TODO: Changing coord systerm in flag to Int + Global Constant definition
show(io::IO, p::Problem) =
    print(io, "MHDFlows Problem\n",
          "  │    Features\n",
          "  |     ├──────────── EOS: ",p.EOS.EOSType,'\n',
          "  │     ├──────── B-field: "*CheckON(p.flag.b),'\n',
          "  ├─────├── spatial Order: "*CheckON(p.flag.SpatialOrder),'\n',
          "  │     ├───── time Order: "*CheckON(p.flag.TimelOrder),'\n',
          "  │     ├───── resolution: ",(p.grid.nx,p.grid.ny,p.grid.nz),'\n',
          "  │     └─────── boundary:  x₁(L/R) ",(p.flag.B_X1L,p.flag.B_X1R),'\n',
          "  │                         x₂(L/R) ",(p.flag.B_X2L,p.flag.B_X2R),'\n',
          "  │                         x₃(L/R) ",(p.flag.B_X3L,p.flag.B_X3R),'\n', 
          "  │     Setting                                            ",'\n',  
          "  │     ├─────────── grid: grid ( $eltype(grid) on GPU)", '\n',
          "  │     ├─────────── flux: flux", '\n',
          "  │     ├── user function: usr_func", '\n',
          "  └─────├─ conserved Vars: sol.U", '\n',
          "        ├─primitived Vars: sol.W", '\n',
          "        └────────── clock: clock", '\n',)
CheckON(Flag_equal_to_True::Bool) = Flag_equal_to_True ? string("ON") : string("OFF");
CheckON(A::Int) = string(A);
