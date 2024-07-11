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
from matplotlib import pyplot as plt
import sys

class complex_bpf():
   def __init__(self, Ntap, Fs_Hz, bandwidth_Hz, centre_freq_Hz):
      self.Ntap = Ntap
      B = bandwidth_Hz/Fs_Hz
      self.alpha = 2*np.pi*centre_freq_Hz/Fs_Hz

      # generate real low pass filter coeffs with bandwidth B/2 Ref: rect_ft.pdf
      self.h = np.zeros(Ntap, dtype=np.csingle)
      for i in range(Ntap):
         n = i-(Ntap-1)/2
         self.h[i] = B*np.sinc(n*B)

      self.mem = np.zeros(self.Ntap-1, dtype=np.csingle)
      self.phase = 1 + 0j

   def bpf(self, x):
      n = len(x)
      phase_vec = self.phase*np.exp(-1j*self.alpha*np.arange(1,n+1))
      x_baseband = x*phase_vec                                         # mix down to baseband
      x_mem = np.concatenate([self.mem,x_baseband])                    # pre-pend filter memory
      x_filt = np.zeros(n, dtype=np.csingle)
      for i in np.arange(n):
         x_filt[i] = np.dot(np.flip(x_mem[i:i+self.Ntap]),self.h)
      self.mem = x_mem[-self.Ntap-1:]                                  # save filter state for next time
      self.phase = phase_vec[-1]                                       # save phase state for next time
      return x_filt*np.conj(phase_vec)                                 # mix back up to centre freq

def complex_bpf_test(plot_en=0):
   Ntap=101
   Fs_Hz = 8000
   bandwidth_Hz = 800
   centre_freq_Hz = 1000
   print(f"Input BPF bandwidth: {bandwidth_Hz:f} centre: {centre_freq_Hz:f}")
   bpf = complex_bpf(Ntap, Fs_Hz, bandwidth_Hz, centre_freq_Hz)

   # -ve freq component of cos() should be attenuated by at least 40dB
   def complex_bpf_test(rx_bpf, pass_str, plot_en):

      Rx_bpf = np.abs(np.fft.fft(rx_bpf*np.hanning(len(rx_bpf))))**2
      power_pos = np.sum(Rx_bpf[:Fs_Hz//2])
      power_neg = np.sum(Rx_bpf[Fs_Hz//2:])
      print(f"power_pos: {power_pos:f} power_neg: {power_neg:f} ratio: {10*np.log10(power_pos/power_neg):f} dB")
      # useful to visualise some plots to debug
      if plot_en:
         plt.figure(1)
         plt.plot(rx_bpf[Ntap-1:].real,rx_bpf[Ntap-1:].imag)
         plt.figure(2)
         plt.plot(10*np.log10(Rx_bpf))
         plt.show()
      if 10*np.log10(power_pos/power_neg) > 40.0:
         print(pass_str)
         return True
      return False         

   # one filtering operation on entire sample
   rx = np.cos(2*np.pi*centre_freq_Hz*np.arange(Fs_Hz)/Fs_Hz)    # 1 sec real sinewave
   rx_bpf = bpf.bpf(rx)
   print(rx.shape,rx_bpf.shape)
   ok1 = complex_bpf_test(rx_bpf[Ntap-1:],"OK1",plot_en)
 
   # test filtering in smaller chunks
   Nmf = 960
   Nframes = len(rx)//Nmf
   rx_bpf2 = np.zeros(0,dtype=np.csingle)
   for f in range(Nframes):
       rx_bpf2 = np.concatenate([rx_bpf2,bpf.bpf(rx[f*Nmf:(f+1)*Nmf])])
   print(Nframes, rx_bpf2.shape)
   ok2 = complex_bpf_test(rx_bpf2[Ntap-1:],"OK2",plot_en)
      
   if ok1 and ok2:
      print("PASS")


class acquisition():
   def __init__(self,Fs,Rs,M,Ncp,Nmf,p,pend,frange=100,fstep=2.5,Pacq_error = 0.0001):
      self.Fs = Fs
      self.Rs = Rs
      self.M = M
      self.Ncp = Ncp
      self.Nmf = Nmf
      self.p = p
      self.pend = pend
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
      
      self.sigma_p = np.sqrt(np.dot(np.conj(p),p).real)
      self.Dtmax12_eoo = 0

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

      candidate = Dtmax12 > Dthresh
     
      self.Dt1 = Dt1
      self.Dt2 = Dt2
      self.Dthresh = Dthresh
      self.Dtmax12 = Dtmax12
      self.f_ind_max = f_ind_max

      return candidate, tmax, fmax
   
   def refine(self, rx, tmax, fmax, ffine_range):
      # TODO: should search over a fine timing range as well, e.g. if we use an 
      # under sampled to save CPU in detect_pilots()
      Fs = self.Fs
      p = self.p
      M = self.M
      Nmf = self.Nmf
   
      Dt1 = np.zeros(len(ffine_range), dtype=np.csingle)
      Dt2 = np.zeros(len(ffine_range), dtype=np.csingle)
      f_ind = 0
      Dtmax = 0

      for f in ffine_range:
         w = 2*np.pi*f/Fs
         # current pilot samples at start of this modem frame
         # TODO should this be using |Dt|?
         w_vec = np.exp(-1j*w*np.arange(M))
         Dt1[f_ind] = np.dot(np.conj(w_vec*rx[tmax:tmax+M]),p)
         # next pilot samples at end of this modem frame
         w_vec = np.exp(-1j*w*(Nmf+np.arange(M)))
         Dt2[f_ind] = np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),p)

         if np.abs(Dt1[f_ind]+Dt2[f_ind]) > Dtmax:
            Dtmax = np.abs(Dt1[f_ind]+Dt2[f_ind])
            fmax = f 
         f_ind = f_ind + 1
      
      self.D_fine = Dt1
      
      return fmax
   
   # spot check using current freq and timing offset.  
   def check_pilots(self, rx, tmax, fmax):
      Fs = self.Fs
      p = self.p
      pend = self.pend
      M = self.M
      Ncp = self.Ncp
      Nmf = self.Nmf

      assert len(rx) == self.Nmf*2+M+Ncp

      # This grid of time-freq Dt samples is already populated by detect_pilots().  Update
      # 5% of the timesteps, so we keep an to date estimate of sigma_r, e.g. if channel noise or
      # signal levels evolve.  This should allow it to adapt over a few seconds, but is a little
      # CPU intensive (e.g. 0.05*960=48 updates/modem frame).
      # TODO: (a) consider alternatives like IIR filter update of sigma_r, (b) compute actual
      # time constant so it's indep of any changes, e.g. in frame rate

      rx_conj = np.conj(rx)
      Nupdate = int(0.05*self.Dt1.shape[0])
      for i in range(Nupdate):
         t = np.random.randint(Nmf)
         self.Dt1[t,:] = np.matmul(rx_conj[t:t+M],self.p_w)
         self.Dt2[t,:] = np.matmul(rx_conj[t+Nmf:t+Nmf+M],self.p_w)

      # Ref: radae.pdf "Pilot Detection over Multiple Frames"
      sigma_r1 = np.mean(np.abs(self.Dt1))/((np.pi/2)**0.5)
      sigma_r2 = np.mean(np.abs(self.Dt2))/((np.pi/2)**0.5)
      sigma_r = (sigma_r1 + sigma_r2)/2.0
      Dthresh = 2*sigma_r*np.sqrt(-np.log(self.Pacq_error/5.0))

      # compare to maxima at current timing and freq offset
      w = 2*np.pi*fmax/Fs
      w_vec = np.exp(-1j*w*np.arange(M))
      Dtmax12 = np.abs(np.dot(np.conj(w_vec*rx[tmax:tmax+M]),p))
      Dtmax12 += np.abs(np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),p))
      valid = Dtmax12 > Dthresh
 
      # compare with end of over sequence
      Dtmax12_eoo = np.abs(np.dot(np.conj(w_vec*rx[tmax+M+Ncp:tmax+2*M+Ncp]),pend))
      Dtmax12_eoo += np.abs(np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),pend))
      endofover = Dtmax12_eoo > Dthresh
     
      self.Dthresh = Dthresh
      self.Dtmax12 = Dtmax12
      self.Dtmax12_eoo = Dtmax12_eoo

      return valid,endofover

# Single modem frame streaming receiver. TODO: is there a better way to pass a bunch of constants around?
class receiver_one():
   def __init__(self,latent_dim,Fs,M,Ncp,Wfwd,Nc,Ns,w,P,bottleneck,pilot_gain,time_offset,coarse_mag):
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
      self.coarse_mag = coarse_mag
      
   # One frame version of do_pilot_eq() for streaming implementation
   def do_pilot_eq_one(self, num_modem_frames, rx_sym_pilots):
      Nc = self.Nc 
      Ns = self.Ns + 1

      # First, estimate the (complex) value of each received pilot symbol
      rx_pilots = torch.zeros(num_modem_frames+1, Nc, dtype=torch.complex64)
      # 3-pilot least squares fit across frequency, ref: freedv_low.pdf
      for i in torch.arange(num_modem_frames+1):
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
      if self.coarse_mag:
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

