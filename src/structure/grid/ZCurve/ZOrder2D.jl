abstract type Tree end

mutable struct Tree2D <: Tree
  LL :: Array{Int,2}
  x0 :: Array{Int,1}
  y0 :: Array{Int,1}
  xD :: Int
  yD :: Int
  id :: Int
end

function Zordering2D!(tree::Tree)  
  xD,yD = tree.xD,tree.yD;
   x, y = 1, 1;
  Zordering!(tree, x, y, xD, yD)  
end

function Zordering!(tree::Tree, x::Int, y::Int, xD::Int, yD::Int)
  # Get the size of the LL
  Lx = size(tree.LL)[1]; 
  Ly = size(tree.LL)[2];
  D  = maximum([xD,yD]);
  Δx = Int(2^(D-1)) .* tree.x0;
  Δy = Int(2^(D-1)) .* tree.y0;
  X = [];
  for (Δxi,Δyi) in zip(Δx,Δy)
    push!(X,(Δxi,Δyi));
  end
  X = unique(X);
  maxΔx = maximum(Δx) - minimum(Δx);
  maxΔy = maximum(Δy) - minimum(Δy);
  if maxΔx > 1 || maxΔy > 1
    for i = 1:length(X)
      xx = Int(x + X[i][1]);
      yy = Int(y + X[i][2]);
      if xx < Lx && yy < Ly
        Zordering!(tree, xx, yy, D-1, D-1);
      elseif xx == Lx && yy == Ly
        fillZ_xy!(tree,xx,yy)
      elseif xx == Lx && yy <= Ly
        fillZ_yD!(tree,xx,yy,D-1)
      elseif yy == Ly && xx <= Lx
        fillZ_xD!(tree,xx,yy,D-1)
      end
    end  
  elseif (x < Lx && y < Ly )
    ZorderFilling!(tree,x,y);
  end
end

function fillZ_xD!(tree::Tree,x::Int,y::Int,xD::Int)
  #Filling the Zoder ID
  x0,y0 = tree.x0,tree.y0;
  Lx,Ly = size(tree.LL);
  #Lx = div(Lx,2) == 0 ? Lx : Lx-1;
  for k = 0:2^(xD)-1
    if x+k <= Lx
      tree.LL[x+k,y] = tree.id + 1;
      tree.id +=1;
    end
  end
end

function fillZ_yD!(tree::Tree,x::Int,y::Int,yD::Int)
  #Filling the Zoder ID
  x0,y0 = tree.x0,tree.y0;
  Lx,Ly = size(tree.LL);
  #Ly = div(Ly,2) == 0 ? Ly : Ly-1;
  for k = 0:2^(yD)-1
    if y + k <= Ly 
      tree.LL[x,y+k] = tree.id + 1;
      tree.id +=1;
    end
  end
end

function fillZ_xy!(tree::Tree,x::Int,y::Int)
  #Filling the Zoder ID
  x0,y0 = tree.x0,tree.y0;
  tree.LL[x,y] = tree.id + 1;
  tree.id +=1;
end

function ZorderFilling!(tree::Tree,x::Int,y::Int)
  #Filling the Zoder ID
  tree.LL[x+tree.x0[1],y+tree.y0[1]] = tree.id + 1;
  tree.LL[x+tree.x0[2],y+tree.y0[2]] = tree.id + 2;
  tree.LL[x+tree.x0[3],y+tree.y0[3]] = tree.id + 3;
  tree.LL[x+tree.x0[4],y+tree.y0[4]] = tree.id + 4;
  tree.id +=4;
end


#test