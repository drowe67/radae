% sig_det
% RADE V2 signal detection using Rayleigh model
% Ry is a ntimestep (rows) by Ncp+M (cols) matrix

function [det,var] = sig_det(Ry)
  beta = 0.95;
  x = 0; x_2 = 0;
  [ntimesteps, cols] = size(Ry);
  for n=1:ntimesteps
    % update EWMA variance est for current timesteps
    x = x*beta + mean(abs(Ry(n,:)))*(1-beta);
    x_2 = x_2*beta + mean(abs(Ry(n,:)).^2)*(1-beta);
    var(n) = x_2 - x^2;
  end  
  det = 0;
endfunction
