"""
Boundary exchange function for Reflactive/Perodic/OutFlow boundary condition
Vars : SwapData : Data in the ghost zone that being swapped in the boundary value exchange process
       TranData : Data in the fluid zone that being snet in the boundary value exchange process

Idea : using the julia array abstraction represention as much as possbile for the future upgrade to GPU and MPI
"""

function X1L_Periodic!(U::CuArray{T,4},grid) where T
    
  Nghost = grid.Nghost::Int;

  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  #exchange of boundary value : 
  SwapData = @view U[      ghost_Lst:ghost_Led,      :,:,:];
  TranData = @view U[end -  bval_Rst:end -  bval_Red,:,:,:];
  copyto!( SwapData, TranData);

  return nothing;
end

function X1R_Periodic!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[end - ghost_Rst:end - ghost_Red,:,:,:];
  TranData = @view U[       bval_Lst:bval_Led       ,:,:,:];
  copyto!( SwapData, TranData);

  return nothing
end

function X2L_Periodic!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  SwapData = @view U[:,       ghost_Lst:ghost_Led      ,:,:];
  TranData = @view U[:, end -  bval_Rst:end -  bval_Red,:,:];
  copyto!( SwapData, TranData);
 
  return nothing;
end

function X2R_Periodic!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;

  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[:,end - ghost_Rst:end - ghost_Red,:,:];
  TranData = @view U[:,       bval_Lst:bval_Led       ,:,:];
  copyto!( SwapData, TranData);
 
  return nothing;
end

function X3L_Periodic!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  SwapData = @view U[:,:,      ghost_Lst:ghost_Led      ,:];
  TranData = @view U[:,:,end -  bval_Rst:end -  bval_Red,:];
  copyto!( SwapData, TranData);

  return nothing
end

function X3R_Periodic!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[:,:,end - ghost_Rst:end - ghost_Red,:];
  TranData = @view U[:,:,       bval_Lst:bval_Led       ,:];
  copyto!( SwapData, TranData);

  return nothing
end


# I need to change Index of the array 
#==================MPI section================================#
function X1L_Periodic_MPI!(U::CuArray{T,4},grid) where T
    
  Nghost = grid.Nghost::Int;

  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  #exchange of boundary value : 
  SwapData = @view U[:,     ghost_Lst:ghost_Led,:,:];
  TranData = @view U[:,end - bval_Rst:end - bval_Red,:,:];

  #dst = target node to send info
  #src = target node to received info
  dst = grid.MPI_info.X1R_Neighbor::Int;
  src = grid.MPI_info.X1L_Neighbor::Int;

  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);

  return nothing;
end

#############
"""
HAVE'T CHANGED YET!!!

"""
#############

function X1R_Periodic_MPI!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[end - ghost_Rst:end - ghost_Red,:,:,:];
  TranData = @view U[       bval_Lst:bval_Led       ,:,:,:];

  #dst = target node to send info
  #src = target node to received info
  dst = grid.MPI_info.X1L_Neighbor::Int;
  src = grid.MPI_info.X1R_Neighbor::Int;

  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);

  return nothing
end

function X2L_Periodic_MPI!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  SwapData = @view U[      ghost_Lst:ghost_Led      ,:,:,:];
  TranData = @view U[end -  bval_Rst:end -  bval_Red,:,:,:];
  
  dst = grid.MPI_info.X2R_Neighbor::Int;
  src = grid.MPI_info.X2L_Neighbor::Int;
  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);
 
  return nothing;
end

function X2R_Periodic_MPI!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;

  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[:,end - ghost_Rst:end - ghost_Red,:,:];
  TranData = @view U[:,       bval_Lst:bval_Led       ,:,:];
  
  dst = grid.MPI_info.X2L_Neighbor::Int;
  src = grid.MPI_info.X2R_Neighbor::Int;
  
  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);
 
  return nothing;
end

function X3L_Periodic_MPI!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  
  ghost_Lst,ghost_Led =            1, Nghost;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;

  SwapData = @view U[:,:,      ghost_Lst:ghost_Led      ,:];
  TranData = @view U[:,:,end -  bval_Rst:end -  bval_Red,:];
  
  dst = grid.MPI_info.X3R_Neighbor::Int;
  src = grid.MPI_info.X3L_Neighbor::Int;
  
  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);

  return nothing
end

function X3R_Periodic_MPI!(U::CuArray{T,4},grid) where T

  Nghost = grid.Nghost::Int;
  ghost_Rst,ghost_Red = Nghost - 1, 0;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;

  SwapData = @view U[:,:,end - ghost_Rst:end - ghost_Red,:];
  TranData = @view U[:,:,       bval_Lst:bval_Led       ,:];

  # node to send data
  dst = grid.MPI_info.X3L_Neighbor::Int;
  # node to recieved dta
  src = grid.MPI_info.X3R_Neighbor::Int;

  MPIDataSwaping!(SwapData, TranData,
                  src,  dst,
                  grid.MPI);

  return nothing
end
