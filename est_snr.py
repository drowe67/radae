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
   p = np.array(model.p)                                    # sequence of pilot samples
   S = np.dot(p,np.conj(p))                                 # signal power/M samples
   g = 1                                                    # channel gain/phase
   N = S/target_SNR                                         # noise power/M samples
   # sequence of noise samples
   sigma = np.sqrt(N/M)/(2**0.5)                            # noise std dev per sample
   n = sigma*(np.random.normal(size=M) + 1j*np.random.normal(size=M))
   N_actual = np.sum(np.conj(n)*n)
   SNR_actual = S/N_actual                                  # will vary slightly from target

   r = g*p + n
   Ct = np.abs(np.dot(np.conj(r),p))**2 / np.dot(np.conj(r),r)
   SNR_est = Ct/(np.dot(np.conj(p),p) - Ct)
   #print(f"S:{S:f} N:{N:f}")
   #print(f"Ct:{Ct:f} SNR_est:{SNR_est:f}")

   return SNR_actual.real, SNR_est.real

# Bring up a RADAE model just to generate the pilot sequence p 
latent_dim = 40
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True)

"""
 TODO 
 1. Estimate over M sample subset when working with CP
 2. Sweep over range of SNRs and check
 3. Consider sweeping over time shifted p to see effect of mis-alignment
 4. Consider running on time shifted p to match timing offset and avoid ISI
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
