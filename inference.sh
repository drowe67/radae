#!/bin/bash -x
#
# Some automation around inference.py to help with testing

OPUS=${HOME}/opus
PATH=${PATH}:${OPUS}

features_in=features.f32
features_out=out.f32

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./inference.sh model in.s16 out.wav [optional inference.py args]"
    echo "usage (play output with aplay):"
    echo "  ./inference.sh model in.s16 - [optional inference.py args]"
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
input_speech=$2
output_speech=$3
features_in=$(mktemp)
features_out=$(mktemp)

# eat first 3 args before passing rest to inference.py in $@
shift; shift; shift

lpcnet_demo -features ${input_speech} ${features_in}
python3 ./inference.py ${model} ${features_in} ${features_out} "$@"
if [ $output_speech == "-" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    aplay $tmp -r 16000 -f S16_LE
else
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    sox -t .s16 -r 16000 -c 1 ${tmp} ${output_speech}
fi
