#!/bin/bash
# Generates plots for pilot based SNR estimator (ref Latex doc SNR Estimtor Attempt 2)
# Run this script, then:
#
# octave:54> radae_plots; est_snr_plot()

# number of time windows to run test over
Nt=60

# straight line correction that is the mean of least squares fit for AWGN/MPG/MPP 
m=0.8070
c=2.513

python3 est_snr.py --eq_ls --Nt $Nt -c $c -m $m --save_text est_snr_awgn.txt
python3 est_snr.py --eq_ls --Nt $Nt -c $c -m $m --h_file h_nc30_mpg.f32 --save_text est_snr_mpg.txt
python3 est_snr.py --eq_ls --Nt $Nt -c $c -m $m --h_file h_nc30_mpp.f32 --save_text est_snr_mpp.txt

