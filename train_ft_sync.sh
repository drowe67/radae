#!/bin/bash -x
#
# Train fine timing and sync networks for a given waveform

OPUS=build/src
PATH=${PATH}:${OPUS}

model_id=250725
model=${model_id}/checkpoints/checkpoint_epoch_200.pth
speech=~/Downloads/all_speech.wav
speech_test=wav/all.wav
inference_args="--rate_Fs --latent-dim 56 --peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128"
train_args=""

function train_fine_timing() {
    ft_model_id=$1
    Nseq=$2
    seq_hop=$3
    if [ ! -f ${model_id}_rx_mpp.f32 ]; then
      ./inference.sh  ${model} ${speech} /dev/null ${inference_args} --write_rx ${model_id}_rx_mpp.f32 --g_file g_mpp.f32
    fi
    #python3 autocorr.py  ${model_id}_rx_mpp.f32 Ry_${ft_model_id}.f32 delta_${ft_model_id}.f32 --Nseq ${Nseq} --seq_hop ${seq_hop} -Q 8 --range_snr --bpf 800 
    #python3 train_ft.py Ry_${ft_model_id}.f32 delta_${ft_model_id}.f32 --epochs 100 --save_model ${model_id}_${ft_model_id}_ft
    python3 train_ft.py Ry_${ft_model_id}.f32 delta_${ft_model_id}.f32 --inference ${model_id}_${ft_model_id}_ft --fte_ml fte_${ft_model_id}_ml.f32 --fte_dsp fte_${ft_model_id}_dsp.f32
}

function train_sync() {
    if [ ! -f ${model_id}_z_train.f32 ]; then
        # "--plot_EqNo" disables actual training, it just runs one epoch to collect results
        python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 200 --lr 0.003 --lr-decay-factor 0.0001 \
        ~/Downloads/tts_speech_16k_speexdsp.f32 tmp --latent-dim 56 --cp 0.004 --auxdata --w1_dec 128 --peak --h_file h_nc14_mpp_train.c64 \
        --h_complex --range_EbNo --range_EbNo_start 3 --timing_rand --freq_rand --ssb_bpf \
        --plot_EqNo ${model_id} --initial-checkpoint ${model} --write_latent ${model_id}_z_train.f32
    fi
    python3 ml_sync.py ${model_id}_z_train.f32 --count 100000 --save_model ${model_id}_ml_sync --latent_dim 56
    python3 ml_sync.py ${model_id}_z_train.f32 --count 100000 --start 1000000 --inference ${model_id}_ml_sync --write_y_hat y_hat.f32 --latent_dim 56
}

function test_ft() {
    ft_model=$1
    snr=$2
    shift; shift
    filename=$(basename -- "${speech_test}")
    filename="${filename%.*}"
    ./inference.sh  ${model} ${speech_test} /dev/null ${inference_args} --write_rx ${filename}_rx.f32 $@
    python3 autocorr.py ${filename}_rx.f32 Ry.f32 delta.f32 --seq_hop 50 -Q 8 --bpf 800 --snr ${snr} --sequence_length 200
    python3 train_ft.py Ry.f32 delta.f32 --inference ${ft_model} --fte_ml fte_ml.f32 --fte_dsp fte_dsp.f32 --sequence_length 200
}

function print_help {
    echo "usage:"
    echo "  ./train_ft_sync.sh mode [options]"
    echo ""
    echo "FT Test mode:"
    echo "  ./train_ft_sync.sh test_ft ft_model snr [options]"
    echo "  ./train_ft_sync.sh test_ft 250725_mpp_16k_ft 10"
    echo "  ./train_ft_sync.sh test_ft 250725_mpp_16k_ft 0 --g_file g_mpp.f32"
    echo ""
    exit 1
}

if [ $# -lt 1 ]; then
    print_help
fi

mode=$1
if [ $mode == "train_ft" ]; then
  if [ $# -lt 2 ]; then
    print_help
  fi
  train_fine_timing $2
fi
if [ $mode == "train_sync" ]; then
  train_sync
fi
if [ $mode == "test_ft" ]; then
  if [ $# -lt 3 ]; then
    print_help
  fi
  ft_model=$2
  snr=$3
  shift; shift; shift
  test_ft $ft_model $snr $@
fi
