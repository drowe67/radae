% sig_det
% RADE V2 signal detection using Rayleigh model
% Ry is a ntimestep (rows) by Ncp+M (cols) matrix

function [delta_hat,det,f_off,Ry_smooth] = sig_det(Ry_norm,Ts=0.42,Fs=8000,M=128)
  alpha = 0.85;
  [ntimesteps, cols] = size(Ry_norm);
  det = zeros(1,ntimesteps);

  Ry_smooth=filter(1-alpha,[1, -alpha],Ry_norm);
  [Ry_max delta_hat] = max(abs(Ry_smooth'));
  det =  Ry_max > Ts;
  f_off = zeros(1,ntimesteps);
  for i=1:ntimesteps
    delta_phi = angle(Ry_smooth(i,delta_hat(i)));
    f_off(i) = -delta_phi*Fs/(2*pi*M);
  end
endfunction
