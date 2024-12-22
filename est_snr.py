"""
/*
  Prototyping code to test estimation of SNR from off air RADAE signals, 
  using pilot statistics.
  
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

def snr_est_test(model, target_EsNo):

   Nc = model.Nc
   P = np.array(model.pilot_gain*model.P) 
   # channel simulation   
   mag = 1 + np.random.rand(1)*99
   g = 1
   tx_sym = P*g
 
   Es = np.dot(tx_sym,np.conj(tx_sym))/Nc
   No = Es/target_EsNo

   # sequence of noise samples
   sigma = np.sqrt(No)/(2**0.5)   
   n = sigma*(np.random.normal(size=Nc) + 1j*np.random.normal(size=Nc))
   rx_sym = g*P + n

   No_actual = np.sum(np.conj(n)*n)/Nc
   EsNo_actual = Es/No_actual

   Ct_sq = np.abs(np.dot(np.conj(rx_sym),P))**2/np.dot(np.conj(rx_sym),rx_sym)
   EsNo_est = Ct_sq/(np.dot(np.conj(P),P) - Ct_sq)

   return EsNo_actual.real, EsNo_est.real

# Bring up a RADAE model just to generate the pilot sequence p 
latent_dim = 80
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004, bottleneck=3)

def single():
   aEsNodB = 10
   EsNo_actual, EsNo_est = snr_est_test(model, 10**(aEsNodB/10))
   print(f"SNR_actual: {10*np.log10(EsNo_actual):5.2f} dB SNR_est: {10*np.log10(EsNo_est):5.2f} dB")


def sweep():
   SNRdB = []
   SNR_estdB = []

   for i in range(25):
      for aSNRdB in np.arange(-5,15):
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

#single()
sweep()
