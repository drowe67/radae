#!/bin/bash -x
# ML EQ experiments - generates sets of curves

epochs=100
n_syms=1000000
batch_size=128
lr=0.1
f_off=2

function train() {
    f=$1
    # train with mse and phase loss functions
    python3 ml_eq.py --frame ${f} --EbNodB 4 --phase_offset --freq_off ${f_off} \
    --lr ${lr} --n_syms ${n_syms} --epochs ${epochs} --batch_size ${batch_size} --save_model mleq0${f}_mse.model --noplots
    python3 ml_eq.py --frame ${f} --loss_phase --EbNodB 4 --phase_offset --freq_off ${f_off} \
    --lr ${lr} --n_syms ${n_syms} --epochs ${epochs} --batch_size ${batch_size} --save_model mleq0${f}_phase.model --noplots
}

function curve() {
    f=$1
    # run BER v Eb/No curves, including DSP lin as control
    python3 ml_eq.py --frame ${f} --notrain --eq dsp --phase_offset --freq_off ${f_off} --curve mleq0${f}_ber_lin.txt
    python3 ml_eq.py --frame ${f} --notrain --load_model mleq0${f}_mse.model --phase_offset --freq_off ${f_off} --curve mleq0${f}_ber_mse.txt 
    python3 ml_eq.py --frame ${f} --notrain --load_model mleq0${f}_phase.model --phase_offset --freq_off ${f_off} --curve mleq0${f}_ber_phase.txt

    # generate EPS plots for paper
    echo "radae_plots; plot_ber_EbNodB( \
    'mleq0${f}_ber_lin.txt','mleq0${f}_ber_mse.txt','mleq0${f}_ber_phase.txt', \
    'mleq0${f}_ber_EbNodB.png','mleq0${f}_ber_EbNodB.eps')" | octave -qf
}

if [ $# -ne 1 ]; then
  echo "usage mleq_cruves.sh framer[1|2]"
  exit 1
fi

#train $1
curve $1

