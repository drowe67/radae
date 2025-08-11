#!/bin/bash -x
#
# Some automation around inference_bbfm.py to help with testing

OPUS=build/src
PATH=${PATH}:${OPUS}

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./bbfm_inference.sh model in.s16 out.wav [optional bbfm_inference.py args]"
    echo "usage (play output with aplay):"
    echo "  ./bbfm_inference.sh model in.s16 - [optional bbfm_inference.py args]"
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
features_in=features_in.f32
features_out=features_out.f32

# eat first 3 args before passing rest to inference.py in $@
shift; shift; shift

lpcnet_demo -features ${input_speech} ${features_in}
python3 ./bbfm_inference.py ${model} ${features_in} ${features_out} "$@" 2> >(tee $bbfm_log >&2)
if [ $? -ne 0 ]; then
  exit 1
fi
if [ $output_speech == "-" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    aplay $tmp -r 16000 -f S16_LE 2>/dev/null
elif [ $output_speech != "/dev/null" ]; then
    tmp=$(mktemp)
    lpcnet_demo -fargan-synthesis ${features_out} ${tmp}
    sox -t .s16 -r 16000 -c 1 ${tmp} ${output_speech}
fi
