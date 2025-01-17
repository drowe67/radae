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

   Rcn_hat_genie = np.abs(h)*Pcn + n
   Ns = model.Ns + 1
   rx_sym_pilots = torch.zeros((1,1,Nw*Ns,Nc), dtype=torch.complex64)
   rx_sym_pilots[0,0,::Ns,:] = torch.tensor(Pcn_hat)
   rx_pilots = receiver.est_pilots(rx_sym_pilots, Nw-1, Nc, Ns)
   rx_pilots = rx_pilots.cpu().detach().numpy()
   rx_phase = np.angle(rx_pilots)
   Rcn_hat_est = Pcn_hat *np.exp(-1j*rx_phase)

   # phase corrected received pilots
   genie_phase = not args.eq_ls

   if genie_phase:
      Rcn_hat = Rcn_hat_genie
   else:
      Rcn_hat = Rcn_hat_est
      
   if args.plots2:
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
   S2_genie = np.sum(np.abs(Rcn_hat_genie.imag)**2)   
   S2_est = np.sum(np.abs(Rcn_hat_est.imag)**2)   
   if genie_phase:      
      snr_est = S1/(2*S2_genie) - 1
   else:
      snr_est = S1/(2*S2_est) - 1
   
   # remove occasional illegal values
   if snr_est <= 0:
      snr_est = 0.1
      
   # actual snr for this time window as check, for AWGN should be
   # close to snr_target, for non untity h it can be quite different
   snr_genie = np.sum(np.abs(h*Pcn)**2)/np.sum(np.abs(n)**2)

   snrdB_genie = 10*np.log10(snr_genie)
   snrdB_est = 10*np.log10(snr_est)
   NdB_genie = 10*np.log10(2*S2_genie)
   NdB_est = 10*np.log10(2*S2_est)

   # user supplied correction factor
   snrdB_est = (snrdB_est - args.c)/args.m

   return snrdB_genie, snrdB_est, NdB_genie, NdB_est

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
   snrdB_genie, snrdB_est, NdB_genie, NdB_est = snr_est_test(model, 10**(snrdB/10), h, Nw, test_S1)
   print(f"setpoint snrdB: {snrdB:5.2f} snrdB_genie: {snrdB_genie:5.2f} snrdB_est: {snrdB_est:5.2f}")

# run over a sequence of timesteps, and return lists of each each est
def sequence(Ntimesteps, snrdB, h, Nw):
   snrdB_genie_list = []
   snrdB_est_list = []
   NdB_genie_list = []
   NdB_est_list = []

   for i in range(Ntimesteps):
      snrdB_genie, snrdB_est, NdB_genie, NdB_est = snr_est_test(model, 10**(snrdB/10), h[i*Nw:(i+1)*Nw,:], Nw)
      
      print(f"setpoint snrdB: {snrdB:5.2f} snrdB_genie: {snrdB_genie:5.2f} snrdB_est: {snrdB_est:5.2f} NdB: {NdB_genie:5.2f} {NdB_est:5.2f}")

      snrdB_genie_list = np.append(snrdB_genie_list, snrdB_genie)
      snrdB_est_list = np.append(snrdB_est_list, snrdB_est)
      NdB_genie_list = np.append(NdB_genie_list, NdB_genie)
      NdB_est_list = np.append(NdB_est_list, NdB_est)
   
   return snrdB_genie_list, snrdB_est_list, NdB_genie,NdB_est

# sweep across SNRs
def sweep(Ntimesteps, h, Nw):
 
   EsNodB_genie = []
   EsNodB_est = []
   NdB_genie = []
   NdB_est = []
   
   r = range(-5,20)
   for aEsNodB in r:
      aEsNodB_genie, aEsNodB_est, aNdB_genie, aNdB_est = sequence(Ntimesteps, aEsNodB, h, Nw)
      EsNodB_genie = np.append(EsNodB_genie, aEsNodB_genie)
      EsNodB_est = np.append(EsNodB_est, aEsNodB_est)
      NdB_genie = np.append(NdB_genie, aNdB_genie)
      NdB_est = np.append(NdB_est, aNdB_est)

   z = np.polyfit(EsNodB_genie, EsNodB_est, 1)
   print(z)
   EsNodB_est_fit = z[0]*EsNodB_genie + z[1]
   
   if args.plots:
      plt.figure(1)
      plt.plot(EsNodB_genie, EsNodB_est,'b+')
      plt.plot(EsNodB_genie, EsNodB_est_fit,'r')
      plt.plot(r,r)
      plt.axis([-5, 20, -5, 20])
      plt.grid()
      plt.xlabel('SNR (dB)')
      plt.ylabel('SNR est (dB)')
  
      """
      z = np.polyfit(NdB_genie, NdB_est, 1)
      print(z)
      NdB_est_fit = z[0]*NdB_genie + z[1]
      print(len(NdB_est))
      plt.figure(2)
      plt.plot(NdB_genie, NdB_est,'b+')
      plt.plot(NdB_genie,NdB_genie)
      plt.plot(NdB_genie, NdB_est_fit,'r')
      plt.grid()
      plt.xlabel('N_genie (dB)')
      plt.ylabel('N_est (dB)')
      """
      plt.show()

   if args.save_text:
      # save test file of test points for Latex plotting in Octave radae_plots.m:est_snr_plot()
      test_points = np.transpose(np.array((EsNodB_genie,EsNodB_est)))
      np.savetxt(args.save_text,test_points,delimiter='\t')

parser = argparse.ArgumentParser()
parser.add_argument('--snrdB', type=float, default=10.0, help='snrdB set point')
parser.add_argument('--single', action='store_true', help='single snrdB test')
parser.add_argument('--sequence', action='store_true', help='run over a sequence of timesteps')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath samples, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('-T', type=float, default=1.0, help='length of time window for estimate (default 1.0 sec)')
parser.add_argument('--Nt', type=int, default=1, help='number of analysis time windows to test across (default 1)')
parser.add_argument('--test_S1', action='store_true', help='calculate S1 two ways to check S1 expression')
parser.add_argument('--eq_ls', action='store_true', help='est phase from received pilots using least square (default genie phase)')
parser.add_argument('--plots', action='store_true', help='sweep results plots (default off)')
parser.add_argument('--plots2', action='store_true', help='debug/internal plots (default off)')
parser.add_argument('--save_text', type=str, default="", help='path to text file to save test points')
parser.add_argument('-c', type=float, default=0.0, help='y offset correction in dB (default 0)')
parser.add_argument('-m', type=float, default=1.0, help='gradient correction in dB (default 1.0)')
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
