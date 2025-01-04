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
from radae import RADAE,receiver_one

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

def snr_est_test(model, snr_target, h, Nw, test_S1=False):

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

   # matrix of received pilots plus noise samples
   Pcn_hat = h*Pcn + n

   # phase corrected received pilots
   genie_phase = not args.eq_ls
   if genie_phase:
      Rcn_hat = np.abs(h)*Pcn + n
   else:
      Ns = model.Ns + 1
      rx_sym_pilots = torch.zeros((1,1,Nw*Ns,Nc), dtype=torch.complex64)
      rx_sym_pilots[0,0,::Ns,:] = torch.tensor(Pcn_hat)
      rx_pilots = receiver.est_pilots(rx_sym_pilots, Nw-1, Nc, Ns)
      rx_pilots = rx_pilots.cpu().detach().numpy()
      rx_phase = np.angle(rx_pilots)
      #print(rx_phase.shape)
      #print(rx_phase)
      Rcn_hat = Pcn_hat *np.exp(-1j*rx_phase)

   if args.plots:
      plt.figure(1)
      plt.plot(Rcn_hat.real, Rcn_hat.imag,'b+')
      plt.figure(2)
      plt.plot(h.real, h.imag,'b+')
      plt.show()

   # calculate S1 two ways to test expression, observe second term is small

   S1 = np.sum(np.abs(Pcn_hat)**2)
   if test_S1:
      S1_first = np.sum(np.abs(h*Pcn)**2)
      S1_second = np.sum(2*(h*Pcn*n).real)
      S1_third = np.sum(np.abs(n)**2)
      S1_sum = S1_first + S1_second + S1_third
      print(f"S1_first: {S1_first:5.2f} S1_second: {S1_second:5.2f} S1_third: {S1_third:5.2f}")
      print(f"S1: {S1:5.2f} S1_sum: {S1_sum:5.2f}")

   # calculate S2 and SNR est
   S2 = np.sum(np.abs(Rcn_hat.imag)**2)   
   snr_est = S1/(2*S2) - 1
 
   # actual snr as check, for AWGN should be close to snr_target, for non untity h 
   # it can be quite different

   snr_check = np.sum(np.abs(h*Pcn)**2)/np.sum(np.abs(n)**2)
   #print(f"S: {np.sum(np.abs(h*Pcn)**2):f} N: {np.sum(np.abs(n)**2)}")
   #print(f"snr:target {snr_target:5.2f} snr_check: {snr_check.real:5.2f} snr_est: {snr_est:5.2f}")

   return snr_est,snr_check

# Bring up a RADAE model
latent_dim = 80
num_features = 20
num_used_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004, bottleneck=3)

# Bring up a receiver instance to use least squares phase est
receiver = receiver_one(model.latent_dim,model.Fs,model.M,model.Ncp,model.Wfwd,model.Nc,
                        model.Ns,model.w,model.P,model.bottleneck,model.pilot_gain,
                        model.time_offset,model.coarse_mag)
print("")

# single timestep test
def single(snrdB, h, Nw, test_S1):
   snr_est, snr_check = snr_est_test(model, 10**(snrdB/10), h, Nw, test_S1)
   print(f"snrdB: {snrdB:5.2f} snrdB_check: {10*np.log10(snr_check):5.2f} snrdB_est: {10*np.log10(snr_est):5.2f}")

# run over a sequence of timesteps, and return lists of each each est
def sequence(Ntimesteps, snrdB, h, Nw):
   snrdB_est_list = []
   snrdB_check_list = []

   for i in range(Ntimesteps):
      snr_est, snr_check = snr_est_test(model, 10**(snrdB/10), h[i*Nw:(i+1)*Nw,:], Nw)
      snrdB_check = 10*np.log10(snr_check)
      snrdB_est = 10*np.log10(snr_est)
      print(f"snrdB: {snrdB:5.2f} snrdB_check: {snrdB_check:5.2f} snrdB_est: {snrdB_est:5.2f}")
      snrdB_est_list = np.append(snrdB_est_list, snrdB_est)
      snrdB_check_list = np.append(snrdB_check_list, snrdB_check)
   
   return  snrdB_est_list, snrdB_check_list

# sweep across SNRs
def sweep(Ntimesteps, h, Nw):
 
   EsNodB_check = []
   EsNodB_est = []
   r = range(-5,20)
   for aEsNodB in r:
      aEsNodB_check, aEsNodB_est = sequence(Ntimesteps, aEsNodB, h, Nw)
      EsNodB_check = np.append(EsNodB_check, aEsNodB_check)
      EsNodB_est = np.append(EsNodB_est, aEsNodB_est)

   plt.figure(1)
   plt.plot(EsNodB_check, EsNodB_est,'b+')
   plt.plot(r,r)
   plt.axis([-5, 20, -5, 20])
   plt.grid()
   plt.xlabel('SNR (dB)')
   plt.ylabel('SNR est (dB)')
   plt.show()

   # save test file of test points for Latex plotting in Octave radae_plots.m:est_snr_plot()
   test_points = np.transpose(np.array((EsNodB_check,EsNodB_est)))
   np.savetxt('est_snr.txt',test_points,delimiter='\t')

parser = argparse.ArgumentParser()
parser.add_argument('--snrdB', type=float, default=10.0, help='snrdB set point')
parser.add_argument('--single', action='store_true', help='single snrdB test')
parser.add_argument('--sequence', action='store_true', help='run over a sequence of timesteps')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath samples, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('-T', type=float, default=1.0, help='length of time window for estimate (default 1.0 sec)')
parser.add_argument('--Nt', type=int, default=1, help='number of analysis time windows to test across (default 1)')
parser.add_argument('--test_S1', action='store_true', help='calculate S1 two ways to check S1 expression')
parser.add_argument('--eq_ls', action='store_true', help='est phase from received pilots usin least square (default genie phase)')
parser.add_argument('--plots', action='store_true', help='debug plots (default off)')
args = parser.parse_args()

Nw = int(args.T // model.Tmf)

if len(args.h_file):
   h = np.fromfile(args.h_file,dtype=np.complex64)
   h = h.reshape((-1,model.Nc))
   # sample once every modem frame
   h = h[::model.Ns+1,:]
   h = h[:Nw*args.Nt,:]
else:
   h = np.ones((Nw*args.Nt,model.Nc))

if args.single:
   single(args.snrdB, h, Nw, args.test_S1)
elif args.sequence:
   sequence(args.Nt, args.snrdB, h, Nw)
else:
   sweep(args.Nt,h, Nw)
