#!/bin/bash -x
#
# Train fine timing and sync networks for a given waveform

OPUS=build/src
PATH=${PATH}:${OPUS}

model_id=250725
model=${model_id}/checkpoints/checkpoint_epoch_200.pth
speech=~/Downloads/all_speech.wav
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

train_fine_timing mpp_4k 4000 25
train_fine_timing mpp_16k 16000 10
#train_sync
