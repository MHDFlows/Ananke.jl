abstract type Tree end
mutable struct Tree3D <: Tree
    LL :: Array{Int,3}
    x0 :: Array{Int,1}
    y0 :: Array{Int,1}
    z0 :: Array{Int,1}
    xD :: Int
    yD :: Int
    zD :: Int
    id :: Int
end

function Zordering3D!(tree::Tree)  
    xD,yD,zD = tree.xD,tree.yD,tree.zD;
    x, y, z = 1,1,1;
    Zordering!(tree, x, y, z, xD, yD, zD);  
end

# In 3D, otherthan 3D Z Curve, we have 7 edge case
# 1. xy 2D plane filling
# 2. yz 2D plane filling
# 3. xy 2D plane filling
# 4. xz 1D line filling
# 5. xy 1D line filling
# 6. yz 1D line filling
# 7. pixel filling in the edge
function Zordering!(tree::Tree, x::Int, y::Int, z::Int, xD::Int, yD::Int, zD::Int)
  # Get the size of the LL
  Lx = size(tree.LL)[1]; 
  Ly = size(tree.LL)[2];
  Lz = size(tree.LL)[3];
  D  = maximum([xD,yD,zD]);
  Δx = Int(2^(D - 1.0)) .* tree.x0;
  Δy = Int(2^(D - 1.0)) .* tree.y0;
  Δz = Int(2^(D - 1.0)) .* tree.z0;
  X = [];
  for (Δxi,Δyi,Δzi) in zip(Δx,Δy,Δz)
    push!(X,(Δxi,Δyi,Δzi));
  end
  X = unique(X);
  maxΔx = maximum(Δx) - minimum(Δx);
  maxΔy = maximum(Δy) - minimum(Δy);
  maxΔz = maximum(Δz) - minimum(Δz);
  if maxΔx > 1 || maxΔy > 1 || maxΔz > 1
    for i = 1:length(X)

      xx = Int(x + X[i][1]);
      yy = Int(y + X[i][2]);
      zz = Int(z + X[i][3]);
      @show xx,yy,zz
      if xx < Lx && yy < Ly && zz < Lz
        Zordering!(tree, xx, yy, zz, D-1, D-1, D-1);
      elseif xx == Lx && yy == Ly && zz == Lz
        fillZ_xyz!(tree,xx,yy,zz)
      elseif xx == Lx && zz == Lz && yy <= Ly
        fillZ_yD!(tree,xx,yy,zz,D-1)
      elseif yy == Ly && zz == Lz && xx <= Lx
        fillZ_xD!(tree,xx,yy,zz,D-1)
      elseif xx == Lx && yy == Ly && zz <= Lz
        fillZ_zD!(tree,xx,yy,zz,D-1)
      elseif xx  < Lx && yy  < Ly && zz == Lz
        # fill XY plane
        fillXYPlane!(tree, xx, yy, zz, D-1, D-1);
      elseif xx  < Lx && yy == Ly && zz  < Lz
        # fill YZ plane
        fillXZPlane!(tree, xx, yy, zz, D-1, D-1);
      elseif xx == Lx && yy  < Ly && zz < Lz
        # fill XZ plane
        fillYZPlane!(tree, xx, yy, zz, D-1, D-1);        
      end
    end
  elseif x < Lx && y < Ly && z < Lz
      ZorderFilling!(tree,x,y,z);
  end
end

# Line Case
function fillZ_xD!(tree::Tree,x::Int,y::Int,z::Int,xD::Int)
  #Filling the Zoder ID
  Lx,Ly,Lz = size(tree.LL);
  for k = 0:2^(xD) - 1
    if x+k <= Lx
      tree.LL[x+k,y,z] = tree.id + 1;
      tree.id +=1;
    end
  end
end

function fillZ_yD!(tree::Tree,x::Int,y::Int,z::Int,yD::Int)
  #Filling the Zoder ID
  Lx,Ly,Lz = size(tree.LL);
  for k = 0:2^(yD) - 1
    if y+k <= Ly
      tree.LL[x,y+k,z] = tree.id + 1;
      tree.id +=1;
    end
  end
end

function fillZ_zD!(tree::Tree,x::Int,y::Int,z::Int,zD::Int)
  #Filling the Zoder ID
  Lx,Ly,Lz = size(tree.LL);
  for k = 0:2^(zD) - 1
    if z+k <= Lz
      tree.LL[x,y,z+k] = tree.id + 1;
      tree.id +=1;
    end
  end
end
function fillZ_xyz!(tree::Tree,x::Int,y::Int,z::Int)
  #Filling the Zoder ID
  tree.LL[x,y,z] = tree.id + 1;
  tree.id +=1;
end

# Plane Case
function fillXZPlane!(tree::Tree, x::Int, y::Int, z::Int, xD::Int, zD::Int);
  # Get the size of the LL
  Lx = size(tree.LL)[1];
  Lz = size(tree.LL)[3];
  Δx = xD > 0 ? Int(2^(xD - 1.0)) .* [0,0,1,1] : 0;
  Δz = zD > 0 ? Int(2^(zD - 1.0)) .* [0,1,0,1] : 0;
  maxΔx = maximum(Δx) - minimum(Δx);
  maxΔz = maximum(Δz) - minimum(Δz);
  if maxΔx > 1 || maxΔz > 1
    xx = x .+ Δx;
    zz = z .+ Δz;
    for i = 1:4
      fillXZPlane!(tree, xx[i], y, zz[i], xD-1, zD-1);
    end
  elseif z < Lz && x < Lx
    for z0 = 0:1, x0=0:1
      tree.LL[x+x0,y,z+z0] = tree.id + 1;
      tree.id += 1;
    end
  end
end

function fillXYPlane!(tree::Tree, x::Int, y::Int, z::Int, xD::Int, yD::Int);
  # Get the size of the LL
  Lx = size(tree.LL)[1]; 
  Ly = size(tree.LL)[2];
  Δx = xD > 0 ? Int(2^(xD - 1.0)) .* [0,0,1,1] : 0;
  Δy = yD > 0 ? Int(2^(yD - 1.0)) .* [0,1,0,1] : 0;
  maxΔx = maximum(Δx) - minimum(Δx);
  maxΔy = maximum(Δy) - minimum(Δy);
  if maxΔx > 1 || maxΔy > 1
    xx = x .+ Δx;
    yy = y .+ Δy;
    for i = 1:4
      fillXYPlane!(tree, xx[i], yy[i], z, xD-1, yD-1);
    end
  elseif maxΔx == maxΔy == 1 && x < Lx && y < Ly
    for y0 = 0:1, x0=0:1
      tree.LL[x+x0,y+y0,z] = tree.id + 1;
      tree.id += 1;
    end
  end
end

function fillYZPlane!(tree::Tree, x::Int, y::Int, z::Int, yD::Int, zD::Int);
  # Get the size of the LL
  Ly = size(tree.LL)[2]; 
  Lz = size(tree.LL)[3];
  Δy = yD > 0 ? Int(2^(yD - 1.0)) .* [0,1,0,1] : 0;
  Δz = zD > 0 ? Int(2^(zD - 1.0)) .* [0,0,1,1] : 0;
  maxΔy = maximum(Δy) - minimum(Δy);
  maxΔz = maximum(Δz) - minimum(Δz);
  if maxΔy > 1 || maxΔz > 1
    yy = y .+ Δy;
    zz = z .+ Δz;
    for i = 1:4
      fillYZPlane!(tree, x, yy[i], zz[i], yD-1, zD-1);
    end
  elseif maxΔy == maxΔz == 1 && y < Ly && z < Lz
    for y0 = 0:1, z0=0:1
      tree.LL[x,y+y0,z+z0] = tree.id + 1;
      tree.id += 1;
    end
  end
end

function ZorderFilling!(tree::Tree,x::Int,y::Int,z::Int)
  #Filling the Zoder ID
  for z0 = 0:1, y0 = 0:1, x0=0:1
    tree.LL[x+x0,y+y0,z+z0] = tree.id + 1;
    tree.id += 1;
  end
end