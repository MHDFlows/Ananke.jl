#module main
include("Time_Integration.jl");
include("utils/UserInterface.jl");

function TimeIntegrator!(prob,t₀ :: Number,N₀ :: Int;
                                       usr_dt = 0.0,
                                     CFL_Coef = 0.25,
                                        diags = [],
                                  loop_number = 100,
                                         save = false,
                                     save_loc = "",
                                     filename = "",
                               FuncWorkInLoop = nothingfunction,
                                  file_number = 0,
                                      dump_dt = 0)

  # Check if save function related parameter
  if (save)
    if length(save_loc) == 0 || length(filename) == 0 || dump_dt == 0
        error("Save Function Turned ON but args save_loc/filename/dump_dt is not declared!\n");
    end 
    file_path_and_name = save_loc*filename;
    savefile(prob, file_number; file_path_and_name = file_path_and_name);
    file_number+=1;
  end

  # Check EOS type & define the CFL function 
  if prob.EOS.EOSType == "Adiabatic"
    UpdateCFL! = Adiabatic.UpdateCFL!
  elseif prob.EOS.EOSType == "Isothermal" 
    UpdateCFL! = Isothermal.UpdateCFL!
  end

  # Set Up the dashboard
  prog = Progress(N₀-prob.clock.step; desc = "Simulation in rogress :", 
                    barglyphs=BarGlyphs('|','█', ['▁' ,'▂' ,'▃' ,'▄' ,'▅' ,'▆', '▇'],' ','|',),
                    barlen=10, showspeed=true)

  # Get the total zone
  Ntotal = prob.grid.nx*prob.grid.ny*prob.grid.nz;
  Step₀  = prob.clock.step;

  prob.clock.dt = usr_dt > 0 ? usr_dt : 0.0;

  time = @elapsed begin 
    while (N₀ >= prob.clock.step) && (t₀ >= prob.clock.t)
      #update the CFL condition;
      if usr_dt == 0.0
        @timeit_debug prob.debugTimer "CFL" CUDA.@sync begin
          UpdateCFL!(prob,prob.sol.W; Coef=0.3, γ = 5/3)
        end
      end
      @timeit_debug prob.debugTimer "Time Stepper" CUDA.@sync begin
        VL2!(prob);
      end
      @timeit_debug prob.debugTimer "User defined function" CUDA.@sync begin
        FuncWorkInLoop(prob);
      end

      prob.clock.step += 1;
      prob.clock.t    += prob.clock.dt;

      Dynamic_Dashboard(prob,prog,N₀,t₀)

    end
  end
  Total_Update_per_second = ( prob.clock.step - Step₀ ) * Ntotal/time/1e6;
  print("Total CPU/GPU time run = $(round(time,digits=3)) s," 
         *"Mzone update per second = $(round(Total_Update_per_second,digits=3)) \n");

  return nothing;
end