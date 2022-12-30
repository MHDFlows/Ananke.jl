#module Datastructure
"Data Structure Module"

#export the main type
export Problem,
	   Equation,
	   Clock,
	   Sol,
	   Flux,
	   EOS,
	   Grid,
	   Ind_struct,
	   xyz_struct

#export the supertype
export AbstractSol,
	     AbstractEOS,
	     AbstractFlux,
	     AbstractGrid,
	     AbstractClock,
	     AbstructEquation

#export subtype
export xyz_struct

# Super Type
abstract type Abstracttmp end
abstract type AbstractEOS end
abstract type AbstractSol end
abstract type AbstractFlux end
abstract type AbstractGrid end
abstract type AbstractClock end
abstract type AbstructFlag end
abstract type AbstructEquation end

# subtype
abstract type xyz_struct end
abstract type MPI_struct end
"Type Problem"
struct Problem{usr_foo,to}
    EOS      :: AbstractEOS #[x]
    grid     :: AbstractGrid #[x]
    flux     :: AbstractFlux #[x]
    sol      :: AbstractSol  #[x]
    clock    :: AbstractClock #[x]
    equation :: AbstructEquation #[x]
    usr_func :: usr_foo
    tmp      :: Abstracttmp
    flag     :: AbstructFlag
  debugTimer :: to
end

"Type Flag"
struct Flag <: AbstructFlag
            "B-field"
            b :: Bool
          "Coordinate System"
        coord :: String
     "The order the time Recostruction"
 SpatialOrder :: Int
     "The order the time integration"
   TimelOrder :: Int
        "Boundary Condition"
        B_X1L :: String
        B_X1R :: String
        B_X2L :: String
        B_X2R :: String
        B_X3L :: String
        B_X3R :: String
end

"Type Equation"
mutable struct Equation <: AbstructEquation
   "The reconstruction function of X₁ dimenion(1st order)"
                 DonorCellX₁! :: Function
   "The reconstruction function of X₂ dimenion(1st order)"
                 DonorCellX₂! :: Function
   "The reconstruction function of X₃ dimenion(1st order)"
                 DonorCellX₃! :: Function
   "The reconstruction function of X₁ dimenion"
   ReconstructionFunction_X₁! :: Function
   "The reconstruction function of X₂ dimenion"
   ReconstructionFunction_X₂! :: Function
   "The reconstruction function of X₃ dimenion"
   ReconstructionFunction_X₃! :: Function
   "The Roe Problem Solver function" 
                   RoeSolver! :: Function
    "The order the time Recostruction"
                 SpatialOrder :: Int
    "The order the time integration"
                   TimelOrder :: Int
    "The boundary value exchange function"
    BoundaryExchangeFunction! :: Function

end

"Type Clock"
mutable struct Clock{T<:AbstractFloat} <: AbstractClock
   "The time"
   t  :: T
   "The time-step"
   dt :: T 
   "The step Number" 
 step :: Int
end

"Tpye Solution"
mutable struct Sol{SolArray} <: AbstractSol
    "Conserved Variables"
    U :: SolArray
    "Primitive Variables"
    W :: SolArray
     "Conserved Variables For Half Step in VL2"
    U_half :: SolArray
end

"Type Flux"
mutable struct Flux{FluxArray} <: AbstractFlux
    F :: FluxArray
    G :: FluxArray
    H :: FluxArray
end

"Type EOS"
mutable struct EOS{T} <: AbstractEOS
    "EOS Type"
    EOSType :: String 
    "As you can see, γ"
    γ :: AbstractFloat 
    "speed of sound"
    cₛ :: T
    "function to transfer Conserved to Primitive"
    ConservedToPrimitive! :: Function 
    "function to transfer Primitive to Conserved"
    PrimitivetoConserved! :: Function 
end


struct Ind_struct
    ρ  :: Int
    e  :: Int
    P  :: Int
    v₁ :: Int
    v₂ :: Int
    v₃ :: Int
    p₁ :: Int
    p₂ :: Int
    p₃ :: Int
end

struct tmp_struct{ReconT}  <: Abstracttmp
    wl  :: ReconT
    wr  :: ReconT
end


Base.eltype(grid::AbstractGrid) = eltype(grid.x1.x1f)
Base.size(prob::Problem) = size(prob.Flux.F) 

#end