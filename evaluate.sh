#!/bin/bash -x
#
# Evaluate a model and sample wave file using AWGN and MPP multipath models, with SSB at same C/No

OPUS=${HOME}/opus
CODEC2=${HOME}/codec2-dev/build_linux/src
PATH=${PATH}:${OPUS}:${CODEC2}

source utils.sh

if [ $# -lt 4 ]; then
    echo "usage:"
    echo "  ./evaluate.sh model sample.[s16|wav] out_dir EbNodB [g_file]"
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
fullfile=$2
out_dir=$3
EbNodB=$4
g_file=$5

# set up command lines to simulate multipath
if [ -f $gfile ]; then
    radae_mp="--g_file ${g_file}"
    cp $g_file fast_fading_samples.float
    ssb_mp="--fading_dir . --mpp"
fi

filename=$(basename -- "$fullfile")
filename="${filename%.*}"

mkdir -p ${out_dir}

# radae simulation
rx=$(mktemp).f32
log=$(./inference.sh ${model} ${fullfile} ${out_dir}/${filename}_${EbNodB}dB.wav \
      --EbNodB ${EbNodB} --write_rx ${rx} --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 $radae_mp)
CNodB=$(echo "$log" | grep "Measured:" | tr -s ' ' | cut -d' ' -f3)

# listen to the modem signal, just keep real channel, filter to simulate listening on a SSB Rx
sox -r 8k -e float -b 32 -c 2 ${rx} -c 1 -e signed-integer -b 16 ${out_dir}/${filename}_${EbNodB}dB_rx.wav sinc 0.3-2.7k remix 1 0
spectrogram "${out_dir}/${filename}_${EbNodB}dB_rx.wav" "${out_dir}/${filename}_${EbNodB}dB_spec.png"

# SSB simulation
speech_8k=$(mktemp).s16
speech_comp=$(mktemp).s16
speech_comp_noise=$(mktemp).s16
sox $fullfile -r 8000 -t .s16 -c 1 $speech_8k
analog_compressor $speech_8k $speech_comp 6

# add noise at same C/No as radae signal, note we measure RMS value after multipath fading if enabled
rms=$(measure_rms $speech_comp $ssb_mp --after_fade)
# 60dB term is due to scaling in ch.c
No=$(python3 -c "import numpy as np; C=10*np.log10(${rms}*${rms}); No=C-${CNodB}-60; print(\"%f\" % No) ")
ch $speech_comp $speech_comp_noise --No ${No} $ssb_mp --after_fade

# adjust peak level to be similar to radae output
radae_peak=$(measure_peak ${out_dir}/${filename}_${EbNodB}dB.wav)
ssb_peak=$(measure_peak $speech_comp_noise)
gain=$(python3 -c "gain=${radae_peak}/${ssb_peak}; print(\"%f\" % gain)")
sox -t .s16 -r 8000 -c 1 -v $gain $speech_comp_noise ${out_dir}/${filename}_${EbNodB}dB_ssb.wav

spectrogram ${out_dir}/${filename}_${EbNodB}dB_ssb.wav ${out_dir}/${filename}_${EbNodB}dB_ssb_spec.png
