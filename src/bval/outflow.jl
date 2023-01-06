"""
Boundary exchange function for Reflactive/Perodic/OutFlow boundary condition
Vars : SwapData : Data in the ghost zone that being swapped in the boundary value exchange process
       TranData : Data in the fluid zone that being snet in the boundary value exchange process

Idea : using the julia array abstraction represention as much as possbile for the future upgrade to GPU and MPI
"""
function X1L_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Lst,ghost_Led =          1,   Nghost
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost
    
  A = grid.x1.BvalA   

  #exchange of boundary value
  SwapData = @view U[ghost_Lst:ghost_Led,:,:,:]
  TranData = @view U[ bval_Lst:bval_Lst ,:,:,:]
  copyto!(SwapData , A.*TranData)

  return nothing
end

function X1R_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Rst,ghost_Red =   Nghost - 1, 0
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost

  A = grid.x1.BvalA  

  SwapData = @view U[end - ghost_Rst:end - ghost_Red, :, :, :]
  TranData = @view U[end -  bval_Red:end -  bval_Red, :, :, :]
  copyto!(SwapData , A.*TranData)

  return nothing
end

function X2L_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Lst,ghost_Led =          1,   Nghost
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost
       
  B = grid.x2.BvalB   
    
  #exchange of boundary value
  SwapData = @view U[:,ghost_Lst:ghost_Led,:,:]
  TranData = @view U[:, bval_Lst:bval_Lst ,:,:]
  copyto!(SwapData , B.*TranData)

  return nothing
end

function X2R_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Rst,ghost_Red =   Nghost - 1, 0
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost
    
  B = grid.x2.BvalB
  
  SwapData = @view U[:,end - ghost_Rst:end - ghost_Red,:,:]
  TranData = @view U[:,end -  bval_Red:end -  bval_Red,:,:]
  copyto!(SwapData , B.*TranData)

  return nothing
end

function X3L_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Lst,ghost_Led =          1,   Nghost
  bval_Lst ,bval_Led  = Nghost + 1, 2*Nghost
         
  C = grid.x3.BvalC::CuArray{T,4}    

  #exchange of boundary value
  SwapData = @view U[:,:,ghost_Lst:ghost_Led,:]
  TranData = @view U[:,:, bval_Lst:bval_Lst ,:]
  copyto!(SwapData , C.*TranData)

  return nothing
end

function X3R_Outflow!(U,grid)
  # Define the ghost zone index and boundary zone index
  Nghost = grid.Nghost::Int
  ghost_Rst,ghost_Red =   Nghost - 1, 0
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost
    
  C = grid.x3.BvalC

  SwapData = @view U[:,:,end - ghost_Rst:end - ghost_Red,:]
  TranData = @view U[:,:,end -  bval_Red:end -  bval_Red,:]
  copyto!(SwapData , C.*TranData)

  return nothing
end


#============== MPI section ========================#

function X1L_Outflow_MPI!(U::CuArray{T,4},grid)  where T
  # MPI version of Outflow
  # In addition to single core outflow, the data in the Right/Left
  #would also need to its neighbor
  
  X1L_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost
  TranData = @view U[end - bval_Rst:end - bval_Red,:,:,:]

  # Get the target node to send my data and send it
  dst = grid.MPI_info.X1R_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)

end

function X1R_Outflow_MPI!(U::CuArray{T,4},grid)  where T
  X1R_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost
  TranData = @view U[bval_Lst:bval_Led ,:,:,:]

  dst = grid.MPI_info.X1L_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)
  
end


function X2L_Outflow_MPI!(U,grid)
  X2L_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost
  TranData = @view U[:,end-bval_Rst:end-bval_Red,:,:]

  # Get the target node to send my data and send it
  dst = grid.MPI_info.X2R_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)
end

function X2R_Outflow_MPI!(U,grid)
  X2R_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost
  TranData = @view U[:,       bval_Lst:bval_Led ,:,:]

  # Get the target node to send my data and send it
  dst = grid.MPI_info.X2L_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)
  
end

function X3L_Outflow_MPI!(U,grid)
  
  X3L_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost
  TranData = @view U[:,:,end -  bval_Rst:end -  bval_Red,:]

  # Get the target node to send my data and send it
  dst = grid.MPI_info.X3R_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)
end

function X3R_Outflow_MPI!(U,grid)
  X3R_Outflow!(U,grid)

  # Get info of the boundary data
  Nghost = grid.Nghost::Int
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost
  TranData = @view U[:,:,bval_Lst:bval_Led,:]

  dst = grid.MPI_info.X3L_Neighbor::Int
  MPIDataSending!(TranData,dst,grid.MPI)
  
end