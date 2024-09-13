#!/bin/bash
#
# The usual wrapper around rx_bbfm.py

OPUS=build/src
PATH=${PATH}:${OPUS}

features_out=features_rx_out.f32

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./rx_bbfm.sh model z_hat.f32 out.wav [optional rx_bbfm.py args]"
    echo "usage (play output with aplay):"
    echo "  ./rx_bbfm.sh model z_hat.f32 - [optional rx_bbfm.py args]"
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

python3 ./rx_bbfm.py ${model} ${input_z_hat} ${features_out} "$@"
if [ $output_speech == "-" ]; then
    lpcnet_demo -fargan-synthesis ${features_out} - | aplay -r 16000 -f S16_LE
elif [ $output_speech != "/dev/null" ]; then
    lpcnet_demo -fargan-synthesis ${features_out} - | sox -t .s16 -r 16000 -c 1 - ${output_speech}
fi
