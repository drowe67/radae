"""
Estimate C/No from a file of samples:
1. Measure C+N in a bandwidth B that contains all the signal power
2. Measure No in an adjacent band that contains just noise
3. Est C = C+N - NoB
4. Est C/No
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('rx', type=str, help='path to signal + noise input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('--window_time', type=float, default=4.0, help='Size of time domain window (seconds) used for SNR measurement.  If sample is longer than this we search for peak in C/No (default 4.0s)')
parser.add_argument('--plots', action='store_true', help='display various plots')
parser.add_argument('--flow', type=float, default=400.0, help='lower freq limit for C+N band (default 400 Hz)')
parser.add_argument('--fhigh', type=float, default=2000.0, help='upper limit for C+N band (default 2000 Hz)')
args = parser.parse_args()

Fs = 8000
B3k = 3000
N = int(Fs*args.window_time)
rx = np.fromfile(args.rx, dtype=np.csingle)
assert len(rx) >= N
max_CNodB = 0
max_time = 0

for st in np.arange(0,len(rx)-N,Fs//4):
    Rx = np.abs(np.fft.fft(rx[st:st+N]))**2
    RxdB = 10*np.log10(Rx)
    bins_per_Hz = N / Fs

    flow_bin = int(bins_per_Hz * args.flow)
    fhigh_bin = int(bins_per_Hz * args.fhigh)
    C_plus_N = np.sum(Rx[flow_bin:fhigh_bin])
    C_plus_N_dB = 10*np.log10(C_plus_N)

    noise_st = fhigh_bin + int(0.1*fhigh_bin)
    noise_en = noise_st + int(0.1*fhigh_bin)

    Nbw = (noise_en-noise_st)/bins_per_Hz
    No = np.sum(Rx[noise_st:noise_en])/Nbw

    C = C_plus_N - No*(args.fhigh-args.flow)
    if C > 0:
        CdB = 10*np.log10(C)
        NodB = 10*np.log10(No)

        CNodB = CdB-NodB
        if CNodB > max_CNodB:
            max_CNodB = CNodB
            max_time = st/Fs

        print(f"time: {st:8d} {st/Fs:5.2f} CNodB: {CNodB:5.2f}")

    if args.plots:
        fig, ax = plt.subplots(1, 1)
        bin_3k = int(bins_per_Hz*B3k)
        x = np.arange(bin_3k)/bins_per_Hz
        mx = np.max(RxdB)
        ax.plot(x, RxdB[:bin_3k])
        ax.plot([args.flow,args.fhigh],[mx,mx],'r')
        ax.plot([int(noise_st/bins_per_Hz),int(noise_en/bins_per_Hz)],[mx,mx],'k')
        plt.show(block=False)
        plt.pause(0.001)
        input("hit[enter] to end.")

max_SNRdB = max_CNodB - 10*np.log10(B3k)
print(f"           Time   C/No    SNR3k")
print(f"Measured: {max_time:5.2f}  {max_CNodB:6.2f}  {max_SNRdB:6.2f}")
