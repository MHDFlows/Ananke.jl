# function for updating dynamical dashboard
function Dynamic_Dashboard(prob,prog,N₀,t₀)
  generate_showvalues(iter) = () -> [(:Progress,iter)];
  n = prob.clock.step;
  t    = round(prob.clock.t,sigdigits=3);
  iter = "iter/Nₒ = $n/$(N₀), t/t₀ = $t/$(t₀)"
  ProgressMeter.next!(prog; showvalues = generate_showvalues(iter));
  return nothing
end