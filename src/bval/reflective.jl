"""
Boundary exchange function for Reflactive/Perodic/OutFlow boundary condition
Vars : SwapData : Data in the ghost zone that being swapped in the boundary value exchange process
       TranData : Data in the fluid zone that being snet in the boundary value exchange process

Idea : using the julia array abstraction represention as much as possbile for the future upgrade to GPU and MPI
"""
function X1L_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Lst,ghost_Led =          1,   Nghost;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;
    
  #exchange of boundary value
  SwapData = @view U[ghost_Led:-1:ghost_Lst,:,:,:];
  TranData = @view U[ bval_Lst: 1:bval_Led ,:,:,:];
  copyto!(SwapData , TranData);

  return nothing;
end


function X1R_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Rst,ghost_Red =   Nghost - 1, 0;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;
     
  SwapData = @view U[:, end - ghost_Red:-1:end - ghost_Rst,:,:];
  TranData = @view U[:, end -  bval_Rst: 1:end -  bval_Red,:,:];
  copyto!(SwapData , TranData);

  return nothing;
end


function X2L_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Lst,ghost_Led =          1,   Nghost;
  bval_Lst,bval_Led   = Nghost + 1, 2*Nghost;
        
  #exchange of boundary value
  SwapData = @view U[:,ghost_Led:-1:ghost_Lst,:,:];
  TranData = @view U[:, bval_Lst: 1:bval_Led ,:,:];
        
  copyto!(SwapData , TranData);

  return nothing;
end


function X2R_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Rst,ghost_Red =   Nghost - 1, 0;
  bval_Rst,bval_Red   = 2*Nghost - 1, Nghost;
  
  SwapData = @view U[:,end - ghost_Red:-1:end - ghost_Rst,:,:];
  TranData = @view U[:,end -  bval_Rst: 1:end -  bval_Red,:,:];
        
  copyto!(SwapData , TranData);

  return nothing
end

function X3L_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Lst,ghost_Led =          1,   Nghost;
  bval_Lst ,bval_Led  = Nghost + 1, 2*Nghost;
    
  #exchange of boundary value
  SwapData = @view U[:,:,ghost_Led:-1:ghost_Lst,:];
  TranData = @view U[:,:, bval_Lst: 1:bval_Led ,:];
        
  copyto!(SwapData , C.*TranData);

  return nothing
end

function X3R_Reflective!(U::CuArray{T,4},grid) where T
  Nghost = grid.Nghost::Int;

  # Define the ghost zone index and boundary zone index
  ghost_Rst,ghost_Red = Nghost-1, 0;
  bval_Rst,bval_Red   = 2*Nghost - 1,   Nghost;
    
  SwapData = @view U[:,:,end - ghost_Red:-1:end - ghost_Rst,:];
  TranData = @view U[:,:,end -  bval_Rst: 1:end -  bval_Red,:];
        
  copyto!(SwapData , TranData);

  return nothing
end