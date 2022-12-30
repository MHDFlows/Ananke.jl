#Boundary Function
# Exchange the boundary function
function BoundaryExchange!(U,grid)
  
  nx,ny,nz = grid.nx::Int,grid.ny::Int,grid.nz::Int;
  x₁,x₂,x₃ = grid.x1,grid.x2,grid.x3;
  if (nx > 1)
    x₁.X₁L_BoundaryExchange!(U,grid);
    x₁.X₁R_BoundaryExchange!(U,grid);
  end
  if (ny > 1)
    x₂.X₂L_BoundaryExchange!(U,grid);
    x₂.X₂R_BoundaryExchange!(U,grid);
  end
  if (nz > 1)
    x₃.X₃L_BoundaryExchange!(U,grid);
    x₃.X₃R_BoundaryExchange!(U,grid);
  end
end