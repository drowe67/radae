% sig_det
% RADE V2 signal detection using Rayleigh model
% Ry is a ntimestep (rows) by Ncp+M (cols) matrix

function [delta_hat,det,f_off,Ry_bar] = sig_det(Ry,Ts=0.42,Fs=8000,M=128)
  alpha = 0.85;
  [ntimesteps, cols] = size(Ry);
  det = zeros(1,ntimesteps);

  Ry_bar=filter(1-alpha,[1, -alpha],Ry);
  [Ry_max delta_hat] = max(abs(Ry_bar'));
  det =  Ry_max > Ts;
  f_off = zeros(1,ntimesteps);
  for i=1:ntimesteps
    delta_phi = angle(Ry_bar(i,delta_hat(i)));
    f_off(i) = -delta_phi*Fs/(2*pi*M);
  end
endfunction
