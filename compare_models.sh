#!/bin/bash
#
# Compare models by plotting loss v Eq/No curves

# run a single training epoch on trained model to build loss versus Eq/No curve
function run_model() {
    model=$1
    dim=$2
    python3 ./train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --latent-dim ${dim} \
        --epochs 1 --lr 0.003 --lr-decay-factor 0.0001 \
        ~/Downloads/tts_speech_16k_speexdsp.f32 tmp \
        --range_EbNo --plot_EqNo ${model} --initial-checkpoint ${model}/checkpoints/checkpoint_epoch_100.pth
}

run_model model05 80  # known good reference
run_model model09 80  # should be a copy of 05
run_model model10 40  # first attempt at dim=40
run_model model11 40  # tanh applied to |z|

echo "radae_plots; loss_EqNo_plot('loss_models','model05_loss_EqNodB.txt','m5','model09_loss_EqNodB.txt', \\
     'm9','model10_loss_EqNodB.txt','m10','model11_loss_EqNodB.txt','m11'); quit" | octave-cli -qf
