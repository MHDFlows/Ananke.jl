include("ZOrder1D.jl");
include("ZOrder2D.jl");
include("ZOrder3D.jl");

function GetRankMap(Nx::Int,Ny::Int,Nz::Int)
  Nᵢ      = findall([Nx,Ny,Nz].!=1);
	ND_Case = length(Nᵢ);
	if ND_Case == 1
	# 1D Case
		LL = Set1DZCurve(Nx,Ny,Nz);
	elseif ND_Case == 2
	# 2D Case
		LL = Set2DZCurve(Nx,Ny,Nz);
	elseif ND_Case == 3
	# 3D Case
		LL = Set3DZCurve(Nx,Ny,Nz);
	end
	return LL
end

finddepth(N::Int) = Int(ceil(log(N)/log(2)));

function Set1DZCurve(Nx::Int,Ny::Int,Nz::Int)

	Nᵢ = findmax([Nx,Ny,Nz])[1];
	LL = Zordering(Nᵢ);
	return reshape(LL,(Nx,Ny,Nz));
end

function Set2DZCurve(Nx::Int,Ny::Int,Nz::Int)
	Ns = [Nx,Ny,Nz];
	N2 = Ns[findall(Ns.!=1)];

	X1 = N2[1];
	X2 = N2[2];

	x0 = [0,1,0,1];
	y0 = [0,0,1,1];
	xD = finddepth(X1);
	yD = finddepth(X2);
	tree = Tree2D(zeros(X1,X2),x0,y0,xD,yD,0);
	Zordering2D!(tree); 
	return reshape(tree.LL,(Nx,Ny,Nz));
end

function Set3DZCurve(Nx::Int,Ny::Int,Nz::Int)

	x0 = [0,1,0,1,0,1,0,1];
	y0 = [0,0,1,1,0,0,1,1];
	z0 = [0,0,0,0,1,1,1,1];
	xD = finddepth(Nx);
	yD = finddepth(Ny);
	zD = finddepth(Nz);
	tree = Tree3D(zeros(Nx,Ny,Nz),x0,y0,z0,xD,yD,zD,0);
	Zordering3D!(tree);  
	return tree.LL;
end