function savefile(prob, file_number; file_path_and_name="")
  space_0 = ""
  for i = 1:4-length(string(file_number));space_0*="0";end
  fw = h5open(file_path_and_name*"_t_"*space_0*string(file_number)*".h5","w")
  
  # point to grid;
  grid = prob.grid;
  
  nx,ny,nz = grid.nx::Int,grid.ny::Int,grid.nz::Int;
  is,ie = grid.x1.is::Int,grid.x1.ie::Int;
  js,je = grid.x2.js::Int,grid.x2.je::Int;
  ks,ke = grid.x3.ks::Int,grid.x3.ke::Int;

  IDN  = grid.ind.ρ::Int
  IVX,IVY,IVZ  = grid.ind.v₁::Int,grid.ind.v₂::Int,grid.ind.v₃::Int;

  @views prim = prob.sol.W;

  # Write density 
  write(fw, "gas_density", Array(view(prim, is:ie, js:je, ks:ke, IDN))

  # Write velocity
  for (nᵢ,ind,i) ∈  zip(( nx, ny, nz),
                        (IVX,IVY,IVZ),
                        ("i","j","k"));
    write(fw, i*"_velocity", Array(view(prim, is:ie, js:je, ks:ke, indᵢ))
  end

  # Write pressure/ speed of sound
  if prob.EOS.EOSType == "Isothermal"
    write(fw, "time", prob.EOS.cₛ);
  else
    IPR = grid.ind.IPR
    write(fw, "time", Array(view(prim, is:ie, js:je, ks:ke, IPR))
  end

  # write time
  write(fw, "time", prob.clock.t);
  close(fw) 
  return nothing
end