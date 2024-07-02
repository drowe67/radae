#!/bin/bash
#
# Some automation around radae_rx.py to help with testing

OPUS=${HOME}/opus
PATH=${PATH}:${OPUS}

features_out=features_rx_out.f32

if [ $# -lt 2 ]; then
    echo "usage (write output to file):"
    echo "  ./radae_rx.sh model rx.iqf32 out.wav [optional radae_rx.py args]"
    echo "usage (play output with aplay):"
    echo "  ./radae_rx.sh model rx.iqf32 - [optional radae_rx.py args]"
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
input_iqf32=$2
output_speech=$3

# eat first 3 args before passing rest to inference.py in $@
shift; shift; shift

cat ${input_iqf32} | python3 radae_rx.py ${model} "$@" >${features_out} 
if [ $output_speech == "-" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    aplay $tmp -r 16000 -f S16_LE
elif [ $output_speech != "/dev/null" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    sox -t .s16 -r 16000 -c 1 ${tmp} ${output_speech}
fi
