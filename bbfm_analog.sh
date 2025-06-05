#!/bin/bash -x
#
# Analog FM simulation, for comparison to ML BBFM

CODEC2_DEV=${CODEC2_DEV:-${HOME}/codec2-dev}
OPUS=build/src
PATH=${PATH}:${OPUS}:${CODEC2_DEV}/build_linux/src:${CODEC2_DEV}/build_linux/misc

which ch >/dev/null || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

source utils.sh

if [ $# -lt 2 ]; then
    echo "usage (write output to file):"
    echo "  ./analog_bbfm.sh in.wav out.wav [--RdBm XX --h_file YY]"
    echo "usage (play output with aplay):"
    echo "  ./analog_bbfm.sh in.wav - [--RdBm XX --h_file YY]"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi

input_speech=$1
output_speech=$2

# eat first 2 args before passing rest to bbfm_analog.py in $@
shift; shift;

#tmp_comp=$(mktemp)
tmp_comp=comp.raw
tmp_out=$(mktemp)
tmp_fm=$(mktemp)

# input wav -> BPF 300-3100 -> pre-emp -> Hilbert compressor -> de-emp -> linearised BBFM sim -> BPF 300-3100 -> output wav

sox ${input_speech} -t .s16 -r 8000 -c 1 - | ch - - --ssbfilt 2 | pre - - | ch - - --clip 8192 --gain 2 --ssbfilt 2 | de - $tmp_comp
cat $tmp_comp | python3 bbfm_analog.py $@ | ch - - --ssbfilt 2 | sox -t .s16 -r 8000 -c 1 - -t .s16 -r 8000 ${tmp_out}

if [ $output_speech == "-" ]; then
    aplay ${tmp_out} -f S16_LE 2>/dev/null
elif [ $output_speech != "/dev/null" ]; then
    sox -t .s16 -r 8000 -c 1 ${tmp_out} -r 8000 ${output_speech}
fi
