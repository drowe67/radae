"""
/*
  Prototyping code to test estimation of SNR from off air RADAE signals, 
  see radio_ae.pdf "SNR Estimation" section.
  
  Copyright (c) 2024 by David Rowe */

/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
"""

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

def snr_est_test(model, target_SNR):

   M = model.M
   p_cp = np.array(model.p_cp)                              # sequence of pilot samples including CP
   time_offset = -16
   p = p_cp[model.Ncp+time_offset:time_offset]              # subset of M samples
   
   # channel simulation   
   phase = np.random.rand(1)*2*np.pi                        # random channel phase
   mag = 1 + np.random.rand(1)*99                           # random channel gain of 1..100
   g = mag*np.exp(1j*phase)                                 # channel gain/phase
   tx = p*g
 
   S = np.dot(tx,np.conj(tx))                               # signal power/M samples
   N = S/target_SNR                                         # noise power/M samples

   # sequence of noise samples
   sigma = np.sqrt(N/M)/(2**0.5)                            # noise std dev per sample
   n = sigma*(np.random.normal(size=M) + 1j*np.random.normal(size=M))
   r = g*p + n
   
   N_actual = np.sum(np.conj(n)*n)
   SNR_actual = S/N_actual                                  # will vary slightly from target

   SNR_est = model.est_snr(torch.tensor(r, dtype=torch.complex64), time_offset)

   return SNR_actual.real, SNR_est.real

# Bring up a RADAE model just to generate the pilot sequence p 
latent_dim = 40
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004)

"""
 TODO 
 1. Consider sweeping over time shifted p to see effect of mis-alignment
 2. Consider running on time shifted p to match timing offset and avoid ISI
"""

SNRdB = []
SNR_estdB = []

for i in range(25):
   for aSNRdB in np.arange(-10,20):
      SNR_actual, SNR_est = snr_est_test(model, 10**(aSNRdB/10))
      SNRdB.append(aSNRdB)
      SNR_estdB.append(10*np.log10(SNR_est))

plt.figure(1)
plt.plot(SNRdB, SNR_estdB,'b+')
plt.grid()
plt.show()

# save test file of test points for Latex plotting in Octave radae_plots.m:est_snr_plot()
test_points = np.transpose(np.array((SNRdB,SNR_estdB)))
np.savetxt('est_snr.txt',test_points,delimiter='\t')
