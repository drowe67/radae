#!/bin/bash
#
# Wrapper around rx2.py

OPUS=build/src
PATH=${PATH}:${OPUS}

features_out=features_out_rx2.f32

if [ $# -lt 4 ]; then
    echo "usage (write output to file):"
    echo "  ./rx2.sh model model_ft model_sync rx.iqf32 out.wav [optional rx.py args]"
    echo "usage (play output with aplay):"
    echo "  ./rx2.sh model model_ft model_sync rx.iqf32 - [optional rx.py args]"
    exit 1
fi
if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi
if [ ! -f $2 ]; then
    echo "can't find $2"
    exit 1
fi
if [ ! -f $3 ]; then
    echo "can't find $3"
    exit 1
fi
if [ ! -f $4 ]; then
    echo "can't find $4"
    exit 1
fi

model=$1
model_ft=$2
model_sync=$3
input_iqf32=$4
output_speech=$5

# eat first 4 args before passing rest to inference.py in $@
shift; shift; shift; shift; shift

python3 ./rx2.py ${model} ${model_ft} ${model_sync} ${input_iqf32} ${features_out} "$@"
if [ $? -ne 0 ]; then
  exit 1
fi
if [ $output_speech == "-" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    aplay $tmp -r 16000 -f S16_LE
elif [ $output_speech != "/dev/null" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    sox -t .s16 -r 16000 -c 1 ${tmp} ${output_speech}
fi
