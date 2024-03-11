import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('rx', type=str, help='path to sine wave + noise input file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('--plots', action='store_true', help='display various plots')
args = parser.parse_args()

# TODO: need a way to read this in from current model
Fs = 8000
Rb = 2000
B = 3000

rx = np.fromfile(args.rx, dtype=np.csingle)
Fs2 = Fs // 2
S = np.abs(np.fft.fft(rx[:Fs]))**2
SdB = 10*np.log10(np.abs(S))

# capture the power from a few bins either side on the max
sine_bin = np.argmax(S[:Fs])
C = np.sum(S[sine_bin-5:sine_bin+5])
CdB = 10*np.log10(C)

noise_st = sine_bin + int(Fs*0.05)
noise_en = sine_bin + int(Fs*0.15)
No = np.mean(S[noise_st:noise_en])
NodB = 10*np.log10(No)

CNodB = CdB-NodB
EbNodB = CNodB - 10*np.log10(Rb)
SNRdB = CNodB - 10*np.log10(B)

print(f"          Eb/No   C/No     SNR3k  Rb")
print(f"Measured: {EbNodB:6.2f}  {CNodB:6.2f}  {SNRdB:6.2f}  {Rb:d}")

if args.plots:
    fig, ax = plt.subplots(1, 1)
    ax.plot(SdB[:Fs2])
    ax.plot(sine_bin,CdB,'rX')
    ax.plot([noise_st,noise_en],[NodB,NodB],'r')
    plt.show(block=False)
    plt.pause(0.001)
    input("hit[enter] to end.")

