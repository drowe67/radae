#!/bin/bash -x
# ML EQ experiments - generates sets of curves

epochs=100

function frame01_train() {
    # train with mse and phase loss functions
    python3 ml_eq.py --EbNodB 4 --phase_offset --lr 0.001 --epochs ${epochs} --save_model mleq01_mse.model --noplots
    python3 ml_eq.py --EbNodB 4 --phase_offset --lr 0.001 --epochs ${epochs} --loss_phase --save_model mleq01_phase.model --noplots
}

function frame01_curve() {
    # run BER v Eb/No curves, including DSP lin as control
    python3 ml_eq.py --eq dsp --notrain --phase_offset --curve mleq01_ber_lin.txt
    python3 ml_eq.py --notrain --curve mleq01_ber_mse.txt --phase_offset --load_model mleq01_mse.model
    python3 ml_eq.py --notrain --curve mleq01_ber_phase.txt --phase_offset --load_model mleq01_phase.model

    # generate EPS plots for paper
    echo "radae_plots; plot_ber_EbNodB( \
    'mleq01_ber_lin.txt','mleq01_ber_mse.txt','mleq01_ber_phase.txt', \
    'mleq01_ber_EbNodB.png','mleq01_ber_EbNodB.eps')" | octave -qf
}

frame01_train
frame01_curve

