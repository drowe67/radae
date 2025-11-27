% sig_det
% RADE V2 signal detection using Rayleigh model
% Ry is a ntimestep (rows) by Ncp+M (cols) matrix

function [det,sigma_r,Ry_bar,Ts] = sig_det(Ry)
  alpha = 0.85; beta = 0.95; Pf=1E-4;
  [ntimesteps, cols] = size(Ry);
  sigma_r = zeros(1,ntimesteps); sigma_r_prev =0;
  Ts = zeros(1,ntimesteps);
  det = zeros(1,ntimesteps);

  % smooth Ry
  Ry_bar=filter(1-alpha,[1, -alpha],Ry);

  Ry_max = max(abs(Ry_bar'));
  for n=1:ntimesteps
    % update EWMA sigma_r est for current timestep
    sigma_r(n) = sigma_r_prev*beta + mean(abs(Ry_bar(n,:)))*(1-beta)/sqrt(pi/2);
    sigma_r_prev = sigma_r(n);
    % use Rayleigh model to see if we have detected a signal
    Ts(n)=sigma_r(n)*sqrt(-2*log(Pf));
    Ts(n) = 0.42;
    det(n) = Ry_max(n) > Ts(n);
  end
endfunction
