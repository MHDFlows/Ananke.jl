module Ananke

import Base: show, summary

using 
  CUDA,
  Statistics,
  DocStringExtensions,
  HDF5,
  FFTW,
  ProgressMeter,
  TimerOutputs


include("Problem_Construction.jl")
include("Integrator.JL")
include("utils/IC.jl")

export 
  ProblemConstrcut,
  TimeIntegrator!,
  SetUpProblemIC!



end