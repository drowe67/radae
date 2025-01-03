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

def snr_est_test(model, snr_target, h, Nw):

   Nc = model.Nc
   Pc = np.array(model.pilot_gain*model.P)

   # matrix of transmitted pilots for time window
   # time steps across rows, carrier across cols
   Pcn = np.zeros((Nw,Nc), dtype=np.complex64)
   for n in np.arange(Nw):
      Pcn[n,:]= Pc

   # sequence of noise samples
   Es = np.sum(Pc**2)/Nc                        # energy per symbol
   No = Es/snr_target                           # noise per symbol
   sigma = np.sqrt(No)/(2**0.5)   
   n = sigma*(np.random.normal(size=(Nw,Nc)) + 1j*np.random.normal(size=(Nw,Nc)))
   print(Es,No,sigma,np.var(n),np.var(Pcn), np.sum(np.abs(Pcn)**2))
   # matrix of received pilots plus noise samples
   Pcn_hat = h*Pcn + n

   # phase corrected received pilots
   Rcn_hat = np.abs(h)*Pcn + n

   # calculate S1 two ways to test expression

   S1 = np.sum(np.abs(Pcn_hat)**2)
   S1_first = np.sum(np.abs(h*Pcn)**2)
   S1_second = np.sum(2*(h*Pcn*n).real)
   #S1_second = np.sum(h*Pcn*np.conj(n) + np.conj(h*Pcn)*n)
   S1_third = np.sum(np.abs(n)**2)
   S1_sum = S1_first + S1_second + S1_third
   print(f"S1_first: {S1_first:5.2f} S1_second: {S1_second:5.2f} S1_third: {S1_third:5.2f}")
   print(f"S1: {S1:5.2f} S1_sum: {S1_sum:5.2f}")

   # calculate SNR est
   S2 = np.sum(np.abs(Rcn_hat.imag)**2)
   print(S1, S2)
   #print(f"S1: {S1:f} S2: {S2:f}")
   
   snr_est = S1/(2*S2) - 1
 
   # actual snr as check, should be same as snr_target
   snr_check = np.sum(np.abs(h*Pcn)**2)/np.sum(np.abs(n)**2)
   print(f"S: {np.sum(np.abs(h*Pcn)**2):f} N: {np.sum(np.abs(n)**2)}")
   print(f"snr:target {snr_target:5.2f} snr_check: {snr_check.real:5.2f} snr_est: {snr_est:5.2f}")

   return snr_est,snr_check

# Bring up a RADAE model
latent_dim = 80
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004, bottleneck=3)

# single timestep test
def single(snrdB, h, Nw):
   snr_est, snr_check = snr_est_test(model, 10**(snrdB/10), h, Nw)
   #print(f"snrdB: {snrdB:5.2f} snrdB_check: {10*np.log10(snr_check):5.2f} snrdB_est: {10*np.log10(snr_est):5.2f}")
   print(f"snrdB: {snrdB:5.2f} snrdB_check: {10*np.log10(snr_check):5.2f} snrdB_est: {10*np.log10(snr_est):5.2f}")

# run over a sequence of timesteps
def sequence(Ntimesteps, EsNodB, h):
   rx_sym = np.zeros((Ntimesteps,model.Nc),dtype=np.csingle)

   print(np.mean(h[:Ntimesteps,:]**2))
   #sum_EsNodB = 0
   #sum_EsNodB_est = 0
   sum_Ct_sq = 0

   for i in range(Ntimesteps):
      Ct_sq, arx_sym = snr_est_test(model, 10**(EsNodB/10), h[i,:])
      #print(f"EsNodB_actual: {10*np.log10(EsNo_actual):5.2f} EsNodB_est: {10*np.log10(EsNo_est):5.2f}")
      #sum_EsNodB += 10*np.log10(EsNo_actual)
      #sum_EsNodB_est += 10*np.log10(EsNo_est)
      sum_Ct_sq += Ct_sq.real
      rx_sym[i,:] = arx_sym
   
   Ct_sq = sum_Ct_sq/Ntimesteps
   P = np.array(model.pilot_gain*model.P) 
   EsNo_est = Ct_sq/(np.dot(np.conj(P),P) - Ct_sq)
   print(Ct_sq, EsNo_est)
   EsNodB_est = 10*np.log10(EsNo_est)
   print(f"EsNodB: {EsNodB:5.2f} EsNodB_est: {EsNodB_est:5.2f}")
 
   return  EsNodB_est, rx_sym

# sweep across SNRs
def sweep(Ntimesteps, h):
 
   EsNodB = []
   EsNodB_est = []
   r = range(-5,15)
   for aEsNodB in r:
      aEsNodB_est, tx_sym = sequence(Ntimesteps, aEsNodB, h)
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
parser.add_argument('--snrdB', type=float, default=10.0, help='snrdB set point')
parser.add_argument('--single', action='store_true', help='single snrdB test')
parser.add_argument('--sequence', action='store_true', help='run over a sequence of timesteps')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath samples, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('-T', type=float, default=1.0, help='length of time window for estimate (default 1.0 sec)')
args = parser.parse_args()

Nw = int(args.T // model.Tmf)

if len(args.h_file):
   h = np.fromfile(args.h_file,dtype=np.float32)
   h = h.reshape((-1,model.Nc))
   print(h.shape)
   # sample once every modem frame
   h = h[::model.Ns+1,:]
   h = h[:Nw,:]
   print(h.shape)
else:
   h = 0.5*np.ones((Nw,model.Nc))
print(h.shape)

if args.single:
   single(args.snrdB,h, Nw)
elif args.sequence:
   snrdB_est, rx_sym = sequence(args.snrdB, h, Nw)
   #print(rx_sym.dtype, rx_sym.shape)

   plt.figure(1)
   plt.plot(rx_sym.real, rx_sym.imag, 'b+')
   mx = np.max(np.abs(rx_sym))
   mx = 10*np.ceil(mx/10)
   plt.axis([-mx,mx,-mx,mx])
   plt.show()
else:
   sweep(args.Nt,h)
