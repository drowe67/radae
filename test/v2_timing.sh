#!/bin/bash -x
#
# RADE V2 Jan 2026 timing investigation script 

OPUS=build/src
PATH=${PATH}:${OPUS}

# larger more realistic for low SNR test
delta=${delta:-0.01}

./inference.sh 250725/checkpoints/checkpoint_epoch_200.pth wav/brian_g8sez.wav /dev/null --rate_Fs --latent-dim 56 \
--peak --cp 0.004 --time_offset -16 --correct_time_offset -16 --auxdata --w1_dec 128 --write_rx 250725_rx.f32

./rx2.sh 250725/checkpoints/checkpoint_epoch_200.pth 250725_ml_sync 250725_rx.f32 /dev/null --latent-dim 56 \
--w1_dec 128 --no_bpf --quiet $@

python3 loss.py features_in.f32 features_out.f32 --features_hat2 features_out_rx2.f32 --clip_start 25
