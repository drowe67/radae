#!/bin/bash -x
#
# Tool to generate a .f32 file of delta_hat samples from a .c64 rx file

rx=$1
delta_hat=$2
shift; shift;

Ry=$(mktemp).c64

python3 autocorr_simple.py ${rx} ${Ry} $@
echo "Ry=load_c64('${Ry}',160); \
      [y,d]=adasmooth(Ry); d = adapp(d); \
      save_f32('${delta_hat}',d)
      quit" | octave-cli -qf 

