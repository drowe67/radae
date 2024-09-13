#!/bin/bash -x
#
# Analog FM simulation, for comparison to ML BBFM

CODEC2_DEV=${CODEC2_DEV:-${HOME}/codec2-dev}
OPUS=build/src
PATH=${PATH}:${OPUS}:${CODEC2_DEV}/build_linux/src
gain=6

which ch >/dev/null || { printf "\n**** Can't find ch - check CODEC2_PATH **** \n\n"; exit 1; }

source utils.sh

if [ $# -lt 3 ]; then
    echo "usage (write output to file):"
    echo "  ./analog_bbfm.sh in.wav out.wav CNRdB"
    echo "usage (play output with aplay):"
    echo "  ./analog_bbfm.sh in.wav - CNRdB"
    exit 1
fi

if [ ! -f $1 ]; then
    echo "can't find $1"
    exit 1
fi

input_speech=$1
output_speech=$2
CNRdB=$3

tmp_in=$(mktemp)
tmp_out=$(mktemp)
tmp_fm=$(mktemp)

# We use hilbert clipper in ch util for speech compressor. Octave FM simulation uses 48 kHz sample rate.
# input wav -> 300-3100Hz Fs=8kHz -> ch compressor -> 300-3100Hz Fs=48kHz -> FM mod/demod
sox ${input_speech} -t .s16 -r 8000 -c 1 - sinc 0.3-3.1k | ch - - --clip 16384 --gain $gain 2>/dev/null | sox -t .s16 -r 8000 -c 1 - -t .s16 -r 48000 ${tmp_in} sinc 0.3-3.1k
echo "fm; pkg load signal; fm_mod_file('${tmp_fm}','${tmp_in}',${CNRdB}); fm_demod_file('${tmp_out}','${tmp_fm}'); quit;" | octave-cli -qf

if [ $output_speech == "-" ]; then
    aplay ${tmp_out} -r 48000 -f S16_LE 2>/dev/null
elif [ $output_speech != "/dev/null" ]; then
    sox -t .s16 -r 48000 -c 1 ${tmp_out} -r 8000 ${output_speech}
fi
