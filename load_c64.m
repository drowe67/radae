% load_c64.m
% David Rowe Oct 2025
%
% load up complex .c64 binary files

function features = load_c64(fn, ncols)
  f=fopen(fn,"rb");
  features_lin=fread(f, 'float32');
  fclose(f);
  size(features_lin)
  assert(length(features_lin) % 2 == 0);
  #features_lin = features_lin(1:2:end) + j*features_lin(2:2:end);
  size(features_lin)
  features_lin(1:2:end);
		 
  nrows = length(features_lin)/ncols;
  features = reshape(features_lin, ncols, nrows);
  features = features.';
endfunction
