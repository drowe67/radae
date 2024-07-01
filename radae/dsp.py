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