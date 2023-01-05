function Get_GlobalMax(SubMax::T, MPI) where T
  Gobal_Max = MPI.Allreduce(SubMax, MPI.MAX, MPI.COMM_WORLD)
  return Gobal_Max
end

function Get_GlobalMin(SubMin::T, MPI) where T
  Gobal_Min = MPI.Allreduce(SubMin, MPI.MIN, MPI.COMM_WORLD)
  return Gobal_Min
end