"""

  Classical DSP support code.

  Copyright (c) 2024 by David Rowe

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

import numpy as np
import torch

def complex_bpf(Ntap, Fs_Hz, bandwidth_Hz, centre_freq_Hz, x):
   B = bandwidth_Hz/Fs_Hz
   alpha = 2*np.pi*centre_freq_Hz/Fs_Hz
   h = np.zeros(Ntap, dtype=np.csingle)

   for i in range(Ntap):
      n = i-(Ntap-1)/2
      h[i] = B*np.sinc(n*B)
   
   x_baseband = x*np.exp(-1j*alpha*np.arange(len(x)))
   x_filt = np.convolve(x_baseband,h)
   return x_filt*np.exp(1j*alpha*np.arange(len(x_filt)))

class acquisition():
   def __init__(self,Fs,Rs,M,Ncp,Nmf,p,frange=100,fstep=2.5,Pacq_error = 0.0001):
      self.Fs = Fs
      self.Rs = Rs
      self.M = M
      self.Ncp = Ncp
      self.Nmf = Nmf
      self.p = p
      self.Pacq_error = Pacq_error
      self.fcoarse_range = np.arange(-frange/2,frange/2,fstep)

      # pre-calculate to speeds things up a bit
      p_w = np.zeros((M,len(self.fcoarse_range)), dtype=np.csingle)
      f_ind = 0
      for f in self.fcoarse_range:
         w = 2*np.pi*f/Fs
         w_vec = np.exp(1j*w*np.arange(M))
         p_w[:,f_ind] = w_vec * p
         f_ind = f_ind + 1
      self.p_w = p_w
      
   def detect_pilots(self, rx):
      Fs = self.Fs
      p = self.p
      M = self.M
      Ncp = self.Ncp
      Nmf = self.Nmf

      # We need a buffer of two and bit modem frames, and search of one Nmf, TODO to reduce
      # latency this could be reduced to a one symbol (M sample) search and Nmf+2*M sample buffer
      assert len(rx) == self.Nmf*2+M+Ncp

      Dt1 = np.zeros((self.Nmf,len(self.fcoarse_range)), dtype=np.csingle)
      Dt2 = np.zeros((self.Nmf,len(self.fcoarse_range)), dtype=np.csingle)
      Dtmax12 = 0
      tmax = 0
      fmax = 0

      # Search modem frame for maxima in correlation between pilots and received signal, over
      # a grid of time and frequency steps.  Note we only correlate on the M samples after the
      # cyclic prefix, so tmax will be Ncp samples after the start of the modem frame

      # TODO: explore strategies to speed up such as under sampled timing, fft for efficient correlation,
      # or ML based acquisition

      rx = np.conj(rx)
      for t in range(Nmf):
         f_ind = 0
         # matrix multiply to speed up calculation of correlation
         # number of cols in first equal to number of rows in second
         Dt1[t,:] = np.matmul(rx[t:t+M],self.p_w)
         Dt2[t,:] = np.matmul(rx[t+Nmf:t+Nmf+M],self.p_w)
         Dt12 = np.abs(Dt1[t,:]) + np.abs(Dt2[t,:])
         local_max = np.max(Dt12)
         if local_max > Dtmax12:
            Dtmax12 = local_max 
            f_ind_max = np.argmax(Dt12)
            fmax = self.fcoarse_range[f_ind_max]
            tmax = t

      # Ref: radae.pdf "Pilot Detection over Multiple Frames"
      sigma_r1 = np.mean(np.abs(Dt1))/((np.pi/2)**0.5)
      sigma_r2 = np.mean(np.abs(Dt2))/((np.pi/2)**0.5)
      sigma_r = (sigma_r1 + sigma_r2)/2.0
      Dthresh = 2*sigma_r*np.sqrt(-np.log(self.Pacq_error/5.0))

      candidate = False
      if Dtmax12 > Dthresh:
         candidate = True
     
      self.Dt1 = Dt1
      self.Dthresh = Dthresh
      self.Dtmax12 = Dtmax12
      self.f_ind_max = f_ind_max

      return candidate, tmax, fmax
   
   def refine(self, rx, tmax, fmax, ffine_range):
      # TODO: should search over a fine timing range as well, e.g. if coarse timing search is under sampled to save CPU
      Fs = self.Fs
      p = self.p
      M = self.M
      Nmf = self.Nmf
   
      D_fine = np.zeros(len(ffine_range), dtype=np.csingle)
      f_ind = 0
      fmax_fine = fmax
      Dtmax = 0

      for f in ffine_range:
         w = 2*np.pi*f/Fs
         # current pilot samples at start of this modem frame
         # TODO should this be using |Dt|?
         w_vec = np.exp(-1j*w*np.arange(M))
         D_fine[f_ind] = np.dot(np.conj(w_vec*rx[tmax:tmax+M]),p)
         # next pilot samples at end of this modem frame
         w_vec = np.exp(-1j*w*(Nmf+np.arange(M)))
         D_fine[f_ind] = D_fine[f_ind] + np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),p)

         if np.abs(D_fine[f_ind]) > Dtmax:
            Dtmax = np.abs(D_fine[f_ind])
            fmax = f 
         f_ind = f_ind + 1

         self.D_fine = D_fine
      return fmax
   

# Single modem frame streaming receiver. TODO: is there a better way to pass a bunch of constnats around?
class receiver_one():
   def __init__(self,latent_dim,Fs,M,Ncp,Wfwd,Nc,Ns,w,P,bottleneck,pilot_gain,time_offset):
      self.latent_dim = latent_dim
      self.Fs = Fs
      self.M = M
      self.Ncp = Ncp
      self.Wfwd = Wfwd
      self.Nc = Nc
      self.Ns = Ns
      self.w = w
      self.P = P
      self.bottleneck = bottleneck
      self.pilot_gain = pilot_gain
      self.time_offset = time_offset

   # One frame version of do_pilot_eq() for streaming implementation
   def do_pilot_eq_one(self, num_modem_frames, rx_sym_pilots):
      Nc = self.Nc 
      Ns = self.Ns + 1

      # First, estimate the (complex) value of each received pilot symbol
      rx_pilots = torch.zeros(num_modem_frames+1, Nc, dtype=torch.complex64)
      # 3-pilot least squares fit across frequency, ref: freedv_low.pdf
      for i in torch.arange(num_modem_frames):
         for c in range(Nc):
               c_mid = c
               # handle edge carriers, alternative is extra "wingman" pilots
               if c == 0:
                  c_mid = 1
               if c == Nc-1:
                  c_mid = Nc-2
               local_path_delay_s = 0.0025      # guess at actual path delay, means a little bit of noise on scatter
               a = local_path_delay_s*self.Fs
               # TODO: I think A & P can be computed off line
               A = torch.tensor([[1, torch.exp(-1j*self.w[c_mid-1]*a)], [1, torch.exp(-1j*self.w[c_mid]*a)], [1, torch.exp(-1j*self.w[c_mid+1]*a)]])
               P = torch.matmul(torch.inverse(torch.matmul(torch.transpose(A,0,1),A)),torch.transpose(A,0,1))
               h = torch.reshape(rx_sym_pilots[0,0,Ns*i,c_mid-1:c_mid+2]/self.P[c_mid-1:c_mid+2],(3,1))
               g = torch.matmul(P,h)
               rx_pilots[i,c] = g[0] + g[1]*torch.exp(-1j*self.w[c]*a)

      # Linearly interpolate between two pilots to EQ data symbol phase
      for i in torch.arange(num_modem_frames):
         for c in torch.arange(0,Nc):
               slope = (rx_pilots[i+1,c] - rx_pilots[i,c])/(self.Ns+1)
               # assume pilots at index 0 and Ns+1, we want to linearly interpolate channel at 1...Ns 
               rx_ch = slope*torch.arange(0,self.Ns+2) + rx_pilots[i,c]
               rx_ch_angle = torch.angle(rx_ch)
               rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]*torch.exp(-1j*rx_ch_angle[1:self.Ns+1])

      # TODO: we may need to average coarse_mag estimate across several frames, especially for multipath channels
      # est RMS magnitude
      mag = torch.mean(torch.abs(rx_pilots)**2)**0.5
      if self.bottleneck == 3:
            mag = mag*torch.abs(self.P[0])/self.pilot_gain
      #print(f"coarse mag: {mag:f}", file=sys.stderr)
      rx_sym_pilots = rx_sym_pilots/mag

      return rx_sym_pilots
   
   #  One frame version of rate Fs receiver for streaming implementation
   def receiver_one(self, rx):
      Ns = self.Ns + 1

      # we expect: Pilots - data symbols - Pilots
      num_timesteps_at_rate_Rs = len(rx) // (self.M+self.Ncp)
      num_modem_frames = num_timesteps_at_rate_Rs // Ns
      assert num_modem_frames == 1
      assert num_timesteps_at_rate_Rs == (Ns+1)

      # remove cyclic prefix
      rx = torch.reshape(rx,(1,num_timesteps_at_rate_Rs,self.M+self.Ncp))
      rx_dash = rx[:,:,self.Ncp+self.time_offset:self.Ncp+self.time_offset+self.M]
      
      # DFT to transform M time domain samples to Nc carriers
      rx_sym = torch.matmul(rx_dash, self.Wfwd)
      
      # Pilot based EQ
      rx_sym_pilots = torch.reshape(rx_sym,(1, num_modem_frames, num_timesteps_at_rate_Rs, self.Nc))
      rx_sym_pilots = self.do_pilot_eq_one(num_modem_frames,rx_sym_pilots)
      rx_sym = torch.ones(1, num_modem_frames, self.Ns, self.Nc, dtype=torch.complex64)
      rx_sym = rx_sym_pilots[:,:,1:self.Ns+1,:]

      # demap QPSK symbols
      rx_sym = torch.reshape(rx_sym, (1, -1, self.latent_dim//2))
      z_hat = torch.zeros(1,rx_sym.shape[1], self.latent_dim)

      z_hat[:,:,::2] = rx_sym.real
      z_hat[:,:,1::2] = rx_sym.imag
      
      return z_hat

