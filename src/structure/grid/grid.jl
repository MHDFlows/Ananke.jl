"Type Grid"
struct Grid{x1_struct,x2_struct,x3_struct,Ind_struct} <: AbstractGrid
    "Size of x,y,z dimensions"
    Nx     :: Int
    Ny     :: Int
    Nz     :: Int
    nx     :: Int
    ny     :: Int
    nz     :: Int
    Nghost :: Int
    Lx_st  :: AbstractFloat
    Ly_st  :: AbstractFloat
    Lz_st  :: AbstractFloat 
    Lx_ed  :: AbstractFloat
    Ly_ed  :: AbstractFloat
    Lz_ed  :: AbstractFloat
    "Information of Face Area & Volume"
    dA₁    :: AbstractFloat
    dA₂    :: AbstractFloat
    dA₃    :: AbstractFloat
    dV     :: AbstractFloat
    "data struct for x,y,z dimenions"
    x1     :: x1_struct
    x2     :: x2_struct
    x3     :: x3_struct
    "data struct for hydro-vars index in 4D array"
    ind    :: Ind_struct
    "Number of total hydro-vars"
    Nhydro :: Int
    NWave  :: Int
end

struct MPI_Info <: MPI_struct
  x1s :: Int
  x1e :: Int
  x2s :: Int
  x2e :: Int
  x3s :: Int
  x3e :: Int
  LL  :: Array
  "src -> target node to send info"
  "dst -> target node to get info"
  X1L_Neighbor :: Int
  X1R_Neighbor :: Int
  X2L_Neighbor :: Int
  X2R_Neighbor :: Int
  X3L_Neighbor :: Int
  X3R_Neighbor :: Int
  MPI
end

struct MPI_Grid{x1_struct,x2_struct,x3_struct,Ind_struct,MPI_struct} <: AbstractGrid
    "Size of x,y,z dimensions"
    Nx     :: Int
    Ny     :: Int
    Nz     :: Int
    nx     :: Int
    ny     :: Int
    nz     :: Int
    Nghost :: Int
    Lx_st  :: AbstractFloat
    Ly_st  :: AbstractFloat
    Lz_st  :: AbstractFloat 
    Lx_ed  :: AbstractFloat
    Ly_ed  :: AbstractFloat
    Lz_ed  :: AbstractFloat
    "Information of Face Area & Volume"
    dA₁    :: AbstractFloat
    dA₂    :: AbstractFloat
    dA₃    :: AbstractFloat
    dV     :: AbstractFloat
    "data struct for x,y,z dimenions"
    x1     :: x1_struct
    x2     :: x2_struct
    x3     :: x3_struct
    "data struct for hydro-vars index in 4D array"
    ind    :: Ind_struct
    "Number of total hydro-vars"
    Nhydro :: Int
    NWave  :: Int
    "MPI information"
    MPI_info :: MPI_struct
end

struct x₁_struct{T1,T4} <: xyz_struct
    is :: Int
    ie :: Int 
    x1f :: T1
    x1v :: T1
    Δx1f :: T1
    Δx1v :: T1
    BvalA :: T4
    X₁L_BoundaryExchange! :: Function
    X₁R_BoundaryExchange! :: Function
end

struct x₂_struct{T1,T4} <: xyz_struct
  js :: Int
  je :: Int 
  x2f :: T1
  x2v :: T1
  Δx2f :: T1
  Δx2v :: T1
  BvalB :: T4
  X₂L_BoundaryExchange! :: Function
  X₂R_BoundaryExchange! :: Function    
end

struct x₃_struct{T1,T4} <: xyz_struct
  ks :: Int
  ke :: Int 
  x3f :: T1
  x3v :: T1
  Δx3f :: T1
  Δx3v :: T1
  BvalC :: T4
  X₃L_BoundaryExchange! :: Function
  X₃R_BoundaryExchange! :: Function
end
