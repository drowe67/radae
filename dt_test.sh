#!/bin/bash -x
#
# Comparing Dtmax/Dthresh for different Nc at same SNR (C/No)

EbNo_18=$1

./inference.sh model17/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB ${EbNo_18} --write_rx rx_17.f32 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3
./rx.sh model17/checkpoints/checkpoint_epoch_100.pth rx_17.f32 /dev/null --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16

EbNo_17=$(python3 -c "EbNo_17=${EbNo_18}+3; print(\"%f\" % EbNo_17) ")
./inference.sh model18/checkpoints/checkpoint_epoch_100.pth wav/brian_g8sez.wav /dev/null --EbNodB ${EbNo_17} --write_rx rx_18.f32 --rate_Fs --pilots --pilot_eq --eq_ls --cp 0.004 --bottleneck 3 --latent-dim 40
./rx.sh model18/checkpoints/checkpoint_epoch_100.pth rx_18.f32 /dev/null --pilots --pilot_eq --bottleneck 3 --cp 0.004 --coarse_mag --time_offset -16 --latent-dim 40

