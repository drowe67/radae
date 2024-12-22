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

def snr_est_test(model, target_EsNo, h):

   Nc = model.Nc
   P = np.array(model.pilot_gain*model.P) 
   tx_sym = P
 
   Es = np.dot(tx_sym,np.conj(tx_sym))/Nc
   No = Es/target_EsNo

   # sequence of noise samples
   sigma = np.sqrt(No)/(2**0.5)   
   n = sigma*(np.random.normal(size=Nc) + 1j*np.random.normal(size=Nc))
   rx_sym = tx_sym*h + n
   #print(P[0],h[0],rx_sym.shape,rx_sym[0])

   No_actual = np.sum(np.conj(n)*n)/Nc
   EsNo_actual = Es/No_actual

   Ct_sq = np.abs(np.dot(np.conj(rx_sym),P))**2/np.dot(np.conj(rx_sym),rx_sym)
   EsNo_est = Ct_sq/(np.dot(np.conj(P),P) - Ct_sq)

   return EsNo_actual.real, EsNo_est.real, rx_sym

# Bring up a RADAE model
latent_dim = 80
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004, bottleneck=3)

# single timestep test
def single(aEsNodB,h):
   EsNo_actual, EsNo_est, rx_sym = snr_est_test(model, 10**(aEsNodB/10), h)
   print(f"EsNodB_actual: {10*np.log10(EsNo_actual):5.2f} EsNodB_est: {10*np.log10(EsNo_est):5.2f}")

# run over a sequence of timesteps
def sequence(Ntimesteps, aEsNodB, h):
   rx_sym = np.zeros((Ntimesteps,model.Nc),dtype=np.csingle)

   print(np.mean(h**2))
   sum_EsNodB = 0
   sum_EsNodB_est = 0

   for i in range(Ntimesteps):
      EsNo_actual, EsNo_est, arx_sym = snr_est_test(model, 10**(aEsNodB/10), h[i,:])
      #print(f"EsNodB_actual: {10*np.log10(EsNo_actual):5.2f} EsNodB_est: {10*np.log10(EsNo_est):5.2f}")
      sum_EsNodB += 10*np.log10(EsNo_actual)
      sum_EsNodB_est += 10*np.log10(EsNo_est)
      rx_sym[i,:] = arx_sym
   
   EsNodB = sum_EsNodB/Ntimesteps
   EsNodB_est = sum_EsNodB_est/Ntimesteps
   print(f"EsNodB_actual: {EsNodB:5.2f} EsNodB_est: {EsNodB_est:5.2f}")
 
   return  EsNodB, EsNodB_est, rx_sym

# sweep across SNRs
def sweep(Ntimesteps, h):
 
   EsNodB = []
   EsNodB_est = []
   r = range(-5,15)
   for aEsNodB in r:
      aEsNodB, aEsNodB_est, tx_sym = sequence(Ntimesteps, aEsNodB, h)
      EsNodB = np.append(EsNodB, aEsNodB)
      EsNodB_est = np.append(EsNodB_est, aEsNodB_est)

   plt.figure(1)
   plt.plot(EsNodB, EsNodB_est,'b+')
   plt.plot(r,r)
   plt.grid()
   plt.show()

   # save test file of test points for Latex plotting in Octave radae_plots.m:est_snr_plot()
   test_points = np.transpose(np.array((EsNodB,EsNodB_est)))
   np.savetxt('est_snr.txt',test_points,delimiter='\t')

parser = argparse.ArgumentParser()
parser.add_argument('--EsNodB', type=float, default=10.0, help='EsNodB set point')
parser.add_argument('--single', action='store_true', help='single EsNodB test')
parser.add_argument('--sequence', action='store_true', help='run over a sequence of timesteps')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath samples, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('--Nt', type=int, default=50, help='Number of timesteps')
args = parser.parse_args()

if len(args.h_file):
   h = np.fromfile(args.h_file,dtype=np.float32)
   h = h.reshape((-1,model.Nc))
else:
   h = np.ones((args.Nt,model.Nc))
print(h.shape)

if args.single:
   single(args.EsNodB,h[0,:])
elif args.sequence:
   EsNodB, EsNodB_est, rx_sym = sequence(args.Nt, args.EsNodB, h)
   #print(rx_sym.dtype, rx_sym.shape)

   plt.figure(1)
   plt.plot(rx_sym.real, rx_sym.imag, 'b+')
   mx = np.max(np.abs(rx_sym))
   mx = 10*np.ceil(mx/10)
   plt.axis([-mx,mx,-mx,mx])
   plt.show()
else:
   sweep(args.Nt,h)
