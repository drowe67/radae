#!/bin/bash -x
#
# Helper script for testing JMVs timing estimator

OPUS=build/src
PATH=${PATH}:${OPUS}

model_id=250725
model=${model_id}/checkpoints/checkpoint_epoch_200.pth
speech=~/Downloads/all_speech.wav
speech_test=wav/all.wav
inference_args="--rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128"
train_args=""

function print_help {
    echo "usage:"
    echo "  ./jmv_ft.sh SNRdB freq_offset [--g_file g_mpp.f32]"
    echo ""
    exit 1
}

if [ $# -lt 1 ]; then
    print_help
fi

snr=$1
freq_off=$2
shift; shift
filename=$(basename -- "${speech_test}")
filename="${filename%.*}"
./inference.sh  ${model} ${speech_test} /dev/null ${inference_args} --write_rx ${filename}_rx.f32 --freq_offset ${freq_off} $@
python3 autocorr.py ${filename}_rx.f32 Ry.c64 delta.f32 --seq_hop 200 -Q 1 --bpf 800 --snr ${snr} --Nseq 3 --sequence_length 1000

ch="awgn"
if [ "$1" == "--g_file" ]; then
  ch="mpp"
fi

png_fn=jmv_ft_${ch}_${snr}dB_${freq_off}Hz.png
png_fn_esc=$(echo ${png_fn} | sed 's/_/\\_/g')

echo "Ry=load_c64('Ry.c64',160); \
      delta=load_f32('delta.f32',1); \
      [y,d]=adasmooth(Ry); \
      set(0, 'defaulttextfontsize', 20); set(0, 'defaultaxesfontsize', 20); \
      figure(2); clf; plot(d,'b;delta\_hat;'); \
      hold on; plot(delta,'r;delta;'); hold off; \
      title('${png_fn_esc}'); \
      print('-dpng','${png_fn}','-S800,600'); \
      quit" | octave-cli -qf 