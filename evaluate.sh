#!/bin/bash
#
# Evaluate a model and sample wave file using AWGN and MPP multipath models, with SSB at same C/No

OPUS=${HOME}/opus
CODEC2=${HOME}/codec2-dev/build_linux/src
PATH=${PATH}:${OPUS}:${CODEC2}

source utils.sh

function print_help {
    echo "usage:"
    echo "  ./evaluate.sh model sample.[s16|wav] out_dir EbNodB [options]"
    echo ""
    echo "Eb/No (dB) is a set point that controls the SNR, adjust by experiment"
    echo
    echo "-d                        Enable debug mode"
    echo "--g_file G_FILE           Simulate multipath channel with rate Fs fading file (default AWGN channel)"
    echo "--bottleneck BOTTLENECK   bottleneck to apply"
    echo "--latent-dim LATENT_DIM   Model z dimension (deafult 80)"
    echo "--peak                    Equalise peak power of RADAE and SSB (default is equal RMS power)"
    exit 1
}

if [ $# -lt 4 ]; then
    print_help
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
shift; shift; shift; shift;
inference_args=""
ch_args=""
channel="awgn"
peak=0

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
    -d)
        set -x	
        shift
    ;;
    --g_file)
        channel="mpp"
        inference_args="${inference_args} --g_file ${2}"	
        cp $g_file fast_fading_samples.float
        ch_args="${ch_args} --fading_dir . --mpp"
        shift
        shift
    ;;
    --latent_dim)
        inference_args="${inference_args} --latent_dim ${2}"	
        shift
        shift
    ;;
   --bottleneck)
        inference_args="${inference_args} --bottleneck ${2}"	
        shift
        shift
    ;;
    --peak)
        peak=1	
        shift
    ;;
    -h)
        print_help	
    ;;
    *)
        print_help
    ;;
esac
done

filename=$(basename -- "$fullfile")
filename="${filename%.*}"

mkdir -p ${out_dir}

# radae simulation
rx=$(mktemp).f32
log=$(./inference.sh ${model} ${fullfile} ${out_dir}/${filename}_${EbNodB}dB_${channel}.wav \
      --EbNodB ${EbNodB} --write_rx ${rx} --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 ${inference_args})
CNodB=$(echo "$log" | grep "Measured:" | tr -s ' ' | cut -d' ' -f3)
PAPRdB=$(echo "$log" | grep "Measured:" | tr -s ' ' | cut -d' ' -f5)
PNodB=$(python3 -c "PNodB=${CNodB}+${PAPRdB}; print(\"%f\" % PNodB) ")

# listen to the modem signal, just keep real channel, filter to simulate listening on a SSB Rx
sox -r 8k -e float -b 32 -c 2 ${rx} -c 1 -e signed-integer -b 16 ${out_dir}/${filename}_${EbNodB}dB_${channel}_rx.wav sinc 0.3-2.7k remix 1 0
spectrogram "${out_dir}/${filename}_${EbNodB}dB_${channel}_rx.wav" "${out_dir}/${filename}_${EbNodB}dB_${channel}_spec.png"

# SSB simulation
speech_8k=$(mktemp).s16
speech_comp=$(mktemp).s16
speech_comp_noise=$(mktemp).s16
sox $fullfile -r 8000 -t .s16 -c 1 $speech_8k
analog_compressor $speech_8k $speech_comp 6

if [ $peak -eq 1 ]; then
  # measure PAPR based on SSB signal before multipath fading applied, as fading messes with PAPR
  papr=$(measure_cpapr $speech_comp)
  # Measure RMS value (after multipath fading if enabled), so we normalise the ups and downs of fading
  rms=$(measure_rms $speech_comp $ch_args --after_fade)
  # Now calculate peak power P required to get target P/No, the 60dB term is due to scaling in ch.c
  No=$(python3 -c "import numpy as np; CdB=10*np.log10(${rms}*${rms}); PdB=CdB+${papr}; No=PdB-${PNodB}-60; print(\"%f\" % No) ")
else
  # add noise at same C/No as radae signal, note we measure RMS value after multipath fading if enabled
  rms=$(measure_rms $speech_comp $ch_args --after_fade)
  # 60dB term is due to scaling in ch.c
  No=$(python3 -c "import numpy as np; CdB=10*np.log10(${rms}*${rms}); No=CdB-${CNodB}-60; print(\"%f\" % No) ")
fi
ch $speech_comp $speech_comp_noise --No ${No} --clip 16384 --after_fade $ch_args

# adjust peak ouput audio level to be similar to radae output
radae_peak=$(measure_peak ${out_dir}/${filename}_${EbNodB}dB_${channel}.wav)
ssb_peak=$(measure_peak $speech_comp_noise)
gain=$(python3 -c "gain=${radae_peak}/${ssb_peak}; print(\"%f\" % gain)")
sox -t .s16 -r 8000 -c 1 -v $gain $speech_comp_noise ${out_dir}/${filename}_${EbNodB}dB_${channel}_ssb.wav

spectrogram ${out_dir}/${filename}_${EbNodB}dB_${channel}_ssb.wav ${out_dir}/${filename}_${EbNodB}dB_${channel}_ssb_spec.png

# reference files and README
cp $fullfile ${out_dir}/zz_${filename}_orig.wav
sox -t .s16 -r 8000 -c 1 $speech_comp ${out_dir}/zz_${filename}_ssb.wav

cat > ${out_dir}/zz_README.txt <<'endreadme'
Radio Autoencoder (radae) samples that demonstrate the system compared to analog SSB

We simulate SSB by compressing the signal, and adjusting the C/No to be the same as the radae signal.  We also adjust the peak level of the Rx "ssb" signal to be about the same as the radae output speech, to make listening convenient. 

General format is filename_EbNodB_channel_proc

filename: the name of the sample e.g. david.wav -> david
CNodB..: for example 30dB
channel.: awgn or mpp (multipath poor)
proc....: suffix describing processing applied to file:
          none: receiver output audio from radio autoencoder
          rx..: the radae modulated received signal, what you would hear "off air" on a SSB receiver before decoding
          spec: spectrogram of "rx" signal
          ssb.: the received SSB "off air" signal, what SSB sounds like at the same C/No.  Compare to "none"
          ssb_spec: The spectrogram of the received SSB signal
zz_filename: The input speech file
zz_filename_ssb: The compressed and bandlimited SSB Tx signal

Use a file manager to present the samples as a matrix of icons, each row at the same Eb/No.  Then click to listen or view spectrograms.

endreadme
