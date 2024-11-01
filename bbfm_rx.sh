#!/bin/bash
#
# The usual wrapper around rx_bbfm.py

OPUS=build/src
PATH=${PATH}:${OPUS}

features_out=features_rx_out.f32

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./bbfm_rx.sh model z_hat.f32 out.wav [optional bbfm_rx.py args]"
    echo "usage (play output with aplay):"
    echo "  ./bbfm_rx.sh model z_hat.f32 - [optional bbfm_rx.py args]"
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

model=$1
input_z_hat=$2
output_speech=$3

# eat first 3 args before passing rest to rx_bbfm.py in $@
shift; shift; shift

python3 ./bbfm_rx.py ${model} ${input_z_hat} ${features_out} "$@"
if [ $? -ne 0 ]; then
  exit 1
fi
if [ $output_speech == "-" ]; then
    lpcnet_demo -fargan-synthesis ${features_out} - | aplay -r 16000 -f S16_LE
elif [ $output_speech != "/dev/null" ]; then
    lpcnet_demo -fargan-synthesis ${features_out} - | sox -t .s16 -r 16000 -c 1 - ${output_speech}
fi
