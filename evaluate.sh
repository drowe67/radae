#!/bin/bash -x
#
# Evaluate a model and sample wave file using various channels model

OPUS=${HOME}/opus
CODEC2=${HOME}/codec2-dev/build_linux/src
PATH=${PATH}:${OPUS}:${CODEC2}

if [ $# -lt 4 ]; then
    echo "usage:"
    echo "  ./evaluate.sh model sample.[s16|wav] out_dir EbNodB"
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

# Approximation of Hilbert clipper type compressor.  
# TODO: modify to make PAPR constant, indep of input level. Best run this on single
# samples rather than files from many speakers with different levels
# TODO: maybe some HF enhancement (but note possibilities are endless here - there is no standard)
function analog_compressor {
    input_file=$1
    output_file=$2
    gain=6
    cat $input_file | ch - - 2>/dev/null | \
    ch - - --No -100 --clip 16384 --gain $gain 2>/dev/null | \
    # final line prints peak and CPAPR for SSB
    ch - - --clip 16384 |
    # manually adjusted to get similar peak levels for SSB and FreeDV
    sox -t .s16 -r 8000 -c 1 -v 0.85 - -t .s16 $output_file
}

model=$1
fullfile=$2
EbNodB=$3
out_dir=$4

filename=$(basename -- "$fullfile")
filename="${filename%.*}"

mkdir -p ${out_dir}
rx=$(mktemp).f32

./inference.sh ${model} ${fullfile} ${out_dir}/${filename}_${EbNodB}dB.wav --EbNodB ${EbNodB} --write_rx ${rx} --rate_Fs
# spectrogram
echo "pkg load signal; rx=load_f32('${rx}',1); plot_specgram(rx, Fs=8000, 0, 2000); print('-dpng','${out_dir}/${filename}_${EbNodB}dB_spec.png'); quit" | octave-cli -qf
# listen to the modem signal
sox -r 8k -e float -b 32 -c 1 ${rx} ${out_dir}/${filename}_${EbNodB}dB_rx.wav sinc 0.3-2.7k

# SSB simulation
speech_8k=$(mktemp).s16
speech_comp=$(mktemp).s16
sox $fullfile -r 8000 -t .s16 -c 1 $speech_8k
analog_compressor $speech_8k $speech_comp
# note experimentally derived conversion to "--No" parameter in ch tool
# TODO: check this is not speech sample specific, we want same SNR (C/No) as radae signal
ch $speech_comp - --No $((-EbNodB-16)) | sox -t .s16 -r 8000 -c 1 - ${out_dir}/${filename}_${EbNodB}dB_ssb.wav

# TODO ssb spectrogram

