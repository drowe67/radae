#!/bin/bash -x
#
# Takes a real valued wave file (e.g. off air samples) and produces
# decoded audio

OPUS=build/src
PATH=${PATH}:${OPUS}

if [ $# -lt 2 ]; then
    echo "usage (write output to file):"
    echo "  ./radae_rx.sh model in.wav out.wav [optional radae_rx.py args]"
    echo "usage (play output with aplay):"
    echo "  ./radae_rx.sh model in.wav - [optional radae_rx.py args]"
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
input_offair=$2
output_speech=$3

# eat first 3 args before passing rest to inference.py in $@
shift; shift; shift

if [ $output_speech == "-" ]; then
    sox ${input_offair} -t .s16 -r 8000 -c 1 - | \
    python3 int16tof32.py --zeropad | \
    python3 radae_rx.py ${model} "$@" | \
    lpcnet_demo -fargan-synthesis - - | \
    aplay $tmp -r 16000 -f S16_LE
elif [ $output_speech != "/dev/null" ]; then
    sox ${input_offair} -t .s16 -r 8000 -c 1 - | \
    python3 int16tof32.py --zeropad | \
    python3 radae_rx.py ${model} "$@" | \
    lpcnet_demo -fargan-synthesis - - | \
    sox -t .s16 -r 16000 -c 1 - ${output_speech}
fi
