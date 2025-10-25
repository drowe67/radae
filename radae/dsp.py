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
import math
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
      
      # if true we can remove time flip from convolution
      assert np.all(self.h == np.flip(self.h))

      self.mem = np.zeros(self.Ntap-1, dtype=np.csingle)
      self.x_mem = np.zeros(self.Ntap-1, dtype=np.csingle)
      self.phase = 1 + 0j
      self.n = -1 

   def bpf(self, x):
      n = len(x)
      phase_vec = self.phase*np.exp(-1j*self.alpha*np.arange(1,n+1))
      x_baseband = x*phase_vec                                         # mix down to baseband
      if n > self.n:
          self.x_filt = np.zeros(n, dtype=np.csingle)
          self.n = n
      if len(self.x_mem) != (len(self.mem) + len(x_baseband)):
          self.x_mem = np.zeros(len(self.mem) + len(x_baseband), dtype=np.csingle)
      np.concatenate([self.mem,x_baseband], out=self.x_mem)                    # pre-pend filter memory
      for i in np.arange(n):
         self.x_filt[i] = np.dot(self.x_mem[i:i+self.Ntap],self.h)
      self.mem = self.x_mem[-self.Ntap-1:]                                  # save filter state for next time
      self.phase = phase_vec[-1]                                       # save phase state for next time
      return self.x_filt[0:n]*np.conj(phase_vec)                            # mix back up to centre freq

def complex_bpf_test(plot_en=0):
   Ntap=101
   Fs_Hz = 8000
   bandwidth_Hz = 800
   centre_freq_Hz = 1000
   print(f"BPF bandwidth: {bandwidth_Hz:f} centre: {centre_freq_Hz:f}")
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
   def __init__(self,Fs,Rs,M,Ncp,Nmf,p,pend,frange=100,fstep=2.5,Pacq_error1 = 0.00001,Pacq_error2 = 0.0001):
      self.Fs = Fs
      self.Rs = Rs
      self.M = M
      self.Ncp = Ncp
      self.Nmf = Nmf
      self.p = p
      self.pend = pend
      self.Pacq_error1 = Pacq_error1
      self.Pacq_error2 = Pacq_error2
      self.fcoarse_range = np.arange(-frange/2,frange/2,fstep)
      self.tfine_range = 0
      self.ffine_range = 0

      self.Dt1 = np.zeros((self.Nmf,len(self.fcoarse_range)), dtype=np.csingle)
      self.Dt2 = np.zeros((self.Nmf,len(self.fcoarse_range)), dtype=np.csingle)
      self.Dt12 = np.zeros((self.Nmf,len(self.fcoarse_range)), dtype=np.csingle)
 
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

      Dtmax12 = 0
      f_ind_max = 0
      tmax = 0
      fmax = 0

      # Search modem frame for maxima in correlation between pilots and received signal, over
      # a grid of time and frequency steps.  Note we only correlate on the M samples after the
      # cyclic prefix, so tmax will be Ncp samples after the start of the modem frame

      # TODO: explore strategies to speed up such as under sampled timing, fft for efficient correlation,
      # or ML based acquisition

      rx = np.conj(rx)
      rx_slided_1 = np.lib.stride_tricks.as_strided(rx[0:], shape=(Nmf,M), strides=rx.strides*2)
      rx_slided_2 = np.lib.stride_tricks.as_strided(rx[Nmf:], shape=(Nmf,M), strides=rx.strides*2)
      np.matmul(rx_slided_1, self.p_w, out=self.Dt1)
      np.abs(self.Dt1, out=self.Dt1)
      np.matmul(rx_slided_2, self.p_w, out=self.Dt2)
      np.abs(self.Dt2, out=self.Dt2)
      np.add(self.Dt1[:], self.Dt2[:], out=self.Dt12)
      maxes = np.max(self.Dt12, axis=1)
      amaxes = np.argmax(self.Dt12, axis=1)
      local_max = np.max(maxes)
      if local_max > Dtmax12:
          Dtmax12 = local_max
          tmax = np.argmax(maxes)
          f_ind_max = amaxes[tmax]
          fmax = self.fcoarse_range[f_ind_max]

      # Ref: radae.pdf "Pilot Detection over Multiple Frames"
      sigma_r1 = np.mean(np.abs(self.Dt1))/((np.pi/2)**0.5)
      sigma_r2 = np.mean(np.abs(self.Dt2))/((np.pi/2)**0.5)
      sigma_r = (sigma_r1 + sigma_r2)/2.0
      Dthresh = 2*sigma_r*np.sqrt(-np.log(self.Pacq_error1/5.0))

      candidate = Dtmax12 > Dthresh
     
      self.Dthresh = Dthresh
      self.Dtmax12 = Dtmax12
      self.f_ind_max = f_ind_max

      return candidate, tmax, fmax
   
   def refine(self, rx, tmax, fmax, tfine_range, ffine_range):
      # TODO: should search over a fine timing range as well, e.g. if we use an 
      # under sampled to save CPU in detect_pilots()
      Fs = self.Fs
      p = self.p
      M = self.M
      Nmf = self.Nmf
 
      if not np.array_equal(tfine_range, self.tfine_range) or not np.array_equal(ffine_range, self.ffine_range): 
          self.Dt1_fine = np.zeros((len(tfine_range),len(ffine_range)), dtype=np.csingle)
          self.Dt2_fine = np.zeros((len(tfine_range),len(ffine_range)), dtype=np.csingle)

      tmax_ind = 0
      Dtmax = 0
      
      f_ind = 0
      for f in ffine_range:
         t_ind = 0
         w = 2*np.pi*f/Fs
         w_vec1 = np.exp(-1j*w*np.arange(M))
         w_vec1_p = w_vec1*np.conj(p)
         w_vec2 = w_vec1*np.exp(-1j*w*Nmf)
         w_vec2_p = w_vec2*np.conj(p)
         for t in tfine_range:
            # current pilot samples at start of this modem frame
            self.Dt1_fine[t_ind,f_ind] = np.dot(rx[t:t+M],w_vec1_p)
            # next pilot samples at end of this modem frame
            self.Dt2_fine[t_ind,f_ind] = np.dot(rx[t+Nmf:t+Nmf+M],w_vec2_p)

            if np.abs(self.Dt1_fine[t_ind,f_ind]+self.Dt2_fine[t_ind,f_ind]) > Dtmax:
               Dtmax = np.abs(self.Dt1_fine[t_ind,f_ind]+self.Dt2_fine[t_ind,f_ind])
               tmax = t
               tmax_ind = t_ind
               fmax = f 
            t_ind = t_ind + 1
         f_ind = f_ind + 1
         
      self.D_fine = self.Dt1_fine[tmax_ind,:]
      
      return tmax, fmax
   
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
         #self.Dt1[t,:] = np.matmul(rx_conj[t:t+M],self.p_w)
         np.matmul(rx_conj[t:t+M],self.p_w, out=self.Dt1[t,:])
         #self.Dt2[t,:] = np.matmul(rx_conj[t+Nmf:t+Nmf+M],self.p_w)
         np.matmul(rx_conj[t+Nmf:t+Nmf+M],self.p_w, out=self.Dt2[t,:])

      # Ref: radae.pdf "Pilot Detection over Multiple Frames"
      sigma_r1 = np.mean(np.abs(self.Dt1))/((np.pi/2)**0.5)
      sigma_r2 = np.mean(np.abs(self.Dt2))/((np.pi/2)**0.5)
      sigma_r = (sigma_r1 + sigma_r2)/2.0
      Dthresh = 2*sigma_r*np.sqrt(-np.log(self.Pacq_error2/5.0))
      Dthresh_eoo = 2*sigma_r*np.sqrt(-np.log(self.Pacq_error1/5.0)) # low thresh of false EOO

      # compare to maxima at current timing and freq offset
      w = 2*np.pi*fmax/Fs
      w_vec = np.exp(-1j*w*np.arange(M))
      Dtmax12 = np.abs(np.dot(np.conj(w_vec*rx[tmax:tmax+M]),p))
      Dtmax12 += np.abs(np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),p))
      valid = Dtmax12 > Dthresh
 
      # compare with end of over sequence
      Dtmax12_eoo = np.abs(np.dot(np.conj(w_vec*rx[tmax+M+Ncp:tmax+2*M+Ncp]),pend))
      Dtmax12_eoo += np.abs(np.dot(np.conj(w_vec*rx[tmax+Nmf:tmax+Nmf+M]),pend))
      endofover = Dtmax12_eoo > Dthresh_eoo
     
      self.Dthresh = Dthresh
      self.Dtmax12 = Dtmax12
      self.Dtmax12_eoo = Dtmax12_eoo

      return valid,endofover

# Single modem frame streaming transmitter.
class transmitter_one():
   def __init__(self,latent_dim,enc_stride,Nzmf,Fs,M,Ncp,Winv,Nc,Ns,w,P,bottleneck,pilot_gain):
      self.latent_dim = latent_dim
      self.enc_stride = enc_stride
      self.Nzmf = Nzmf
      self.Fs = Fs
      self.M = M
      self.Ncp = Ncp
      self.Winv = Winv
      self.Nc = Nc
      self.Ns = Ns
      self.w = w
      self.P = P
      self.bottleneck = bottleneck
      self.pilot_gain = pilot_gain
      
   # One frame version of rate Fs transmitter for streaming implementation
   def transmitter_one(self, z, num_timesteps_at_rate_Rs):
      tx_sym = z[:,:,::2] + 1j*z[:,:,1::2]

      # constrain magnitude of 2D complex symbols 
      if self.bottleneck == 2:
         tx_sym = torch.tanh(torch.abs(tx_sym))*torch.exp(1j*torch.angle(tx_sym))
         
      # reshape into sequence of OFDM modem frames
      tx_sym = torch.reshape(tx_sym,(1,num_timesteps_at_rate_Rs,self.Nc))

      # insert pilot symbols at the start of each modem frame
      num_modem_frames = num_timesteps_at_rate_Rs // self.Ns
      tx_sym = torch.reshape(tx_sym,(1, num_modem_frames, self.Ns, self.Nc))
      tx_sym_pilots = torch.zeros(1, num_modem_frames, self.Ns+1, self.Nc, dtype=torch.complex64,device=tx_sym.device)
      tx_sym_pilots[:,:,1:self.Ns+1,:] = tx_sym
      tx_sym_pilots[:,:,0,:] = self.pilot_gain*self.P
      num_timesteps_at_rate_Rs = num_timesteps_at_rate_Rs + num_modem_frames
      tx_sym = torch.reshape(tx_sym_pilots,(1, num_timesteps_at_rate_Rs, self.Nc))

      num_timesteps_at_rate_Fs = num_timesteps_at_rate_Rs*self.M

      # IDFT to transform Nc carriers to M time domain samples
      tx = torch.matmul(tx_sym, self.Winv)

      # Optionally insert a cyclic prefix
      Ncp = self.Ncp
      if self.Ncp:
            tx_cp = torch.zeros((1,num_timesteps_at_rate_Rs,self.M+Ncp),dtype=torch.complex64,device=tx.device)
            tx_cp[:,:,Ncp:] = tx
            tx_cp[:,:,:Ncp] = tx_cp[:,:,-Ncp:]
            tx = tx_cp
            num_timesteps_at_rate_Fs = num_timesteps_at_rate_Rs*(self.M+Ncp)
      tx = torch.reshape(tx,(1,num_timesteps_at_rate_Fs))                         
      
      # Constrain magnitude of complex rate Fs time domain signal, simulates Power
      # Amplifier (PA) that saturates at abs(tx) ~ 1
      if self.bottleneck == 3:
         tx = torch.tanh(torch.abs(tx)) * torch.exp(1j*torch.angle(tx))
      return tx



# Single modem frame streaming receiver. TODO: is there a better way to pass a bunch of constants around?
class receiver_one():
   def __init__(self,latent_dim,Fs,M,Ncp,Wfwd,Nc,Ns,w,P,Pend,bottleneck,pilot_gain,time_offset,coarse_mag):
      self.latent_dim = latent_dim
      self.Fs = Fs
      self.M = M
      self.Ncp = Ncp
      self.Wfwd = Wfwd
      self.Nc = Nc
      self.Ns = Ns
      self.w = w
      self.P = P
      self.Pend = Pend
      self.bottleneck = bottleneck
      self.pilot_gain = pilot_gain
      self.time_offset = time_offset
      self.coarse_mag = coarse_mag

      # pre-compute some matrices
      self.Pmat = torch.empty(Nc,2,3,dtype=torch.complex64)
      for c in range(Nc):
         c_mid = c
         # handle edge carriers, alternative is extra "wingman" pilots
         if c == 0:
            c_mid = 1
         if c == Nc-1:
            c_mid = Nc-2
         local_path_delay_s = 0.0025      # guess at actual path delay, means a little bit of noise on scatter
         a = local_path_delay_s*self.Fs
         A = torch.tensor([[1, torch.exp(-1j*self.w[c_mid-1]*a)], [1, torch.exp(-1j*self.w[c_mid]*a)], [1, torch.exp(-1j*self.w[c_mid+1]*a)]])
         self.Pmat[c] = torch.matmul(torch.inverse(torch.matmul(torch.transpose(A,0,1),A)),torch.transpose(A,0,1))

      self.snrdB_3k_est = 0
      self.m = 0.8070
      self.c = 2.513
         
   def est_pilots(self, rx_sym_pilots, num_modem_frames, Nc, Ns):
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
               h = torch.reshape(rx_sym_pilots[0,0,Ns*i,c_mid-1:c_mid+2]/self.P[c_mid-1:c_mid+2],(3,1))
               g = torch.matmul(self.Pmat[c],h)
               rx_pilots[i,c] = g[0] + g[1]*torch.exp(-1j*self.w[c]*a)

      return rx_pilots

   # update SNR estimate
   def update_snr_est(self, rx_sym_pilots, rx_pilots):
      Pcn_hat = rx_sym_pilots[0,0,0,:]
      rx_phase = torch.angle(rx_pilots[0,:])
      Rcn_hat = Pcn_hat * torch.exp(-1j*rx_phase)
      S1 = torch.sum(torch.abs(Pcn_hat)**2)
      S2 = torch.sum(torch.abs(Rcn_hat.imag)**2) + 1E-12 
      snr_est = S1/(2*S2) - 1
      # remove occasional illegal values
      if snr_est <= 0:
         snr_est = 0.1
      snrdB_est = 10*np.log10(snr_est)
      # correction based on average of straight line fit to AWGN/MPG/MPP
      snrdB_est = (snrdB_est - self.c)/self.m
      # convert to 3000Hz noise badnwidth, and account for carrier power in cyclic prefix
      Rs = self.Fs/self.M
      snrdB_3k_est = snrdB_est + 10*math.log10(Rs*self.Nc/3000) + 10*math.log10((self.M+self.Ncp)/self.M)

      # moving average smoothing, roughly 1 second time constant
      self.snrdB_3k_est = 0.9*self.snrdB_3k_est + 0.1*snrdB_3k_est
      
   # One frame version of do_pilot_eq() for streaming implementation
   def do_pilot_eq_one(self, num_modem_frames, rx_sym_pilots):
      Nc = self.Nc 
      Ns = self.Ns + 1

      # First, estimate the (complex) value of each received pilot symbol, and update SNR est
      rx_pilots = self.est_pilots(rx_sym_pilots, num_modem_frames, Nc, Ns)
      self.update_snr_est(rx_sym_pilots, rx_pilots)
      
      # Linearly interpolate between two pilots to EQ data symbol phase
      for i in torch.arange(num_modem_frames):
         for c in torch.arange(0,Nc):
               slope = (rx_pilots[i+1,c] - rx_pilots[i,c])/(self.Ns+1)
               # assume pilots at index 0 and Ns+1, we want to linearly interpolate channel at 1...Ns 
               rx_ch = slope*torch.arange(0,self.Ns+2) + rx_pilots[i,c]
               rx_ch_angle = torch.angle(rx_ch)
               rx_sym_pilots[0,i,1:self.Ns+1,c] = rx_sym_pilots[0,i,1:self.Ns+1,c]*torch.exp(-1j*rx_ch_angle[1:self.Ns+1])

      # est ampl across one just two sets of pilots seems to work OK (loss isn't impacted)
      if self.coarse_mag:
         mag = torch.mean(torch.abs(rx_pilots)**2)**0.5 + 1E-6
         if self.bottleneck == 3:
            mag = mag*torch.abs(self.P[0])/self.pilot_gain
         #print(f"coarse mag: {mag:f}", file=sys.stderr)
         rx_sym_pilots = rx_sym_pilots/mag

      return rx_sym_pilots
   
   #  One frame version of rate Fs receiver for streaming implementation
   def receiver_one(self, rx, endofover):
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
      
      rx_sym_pilots = torch.reshape(rx_sym,(1, num_modem_frames, num_timesteps_at_rate_Rs, self.Nc))
      if not endofover:
         # Pilot based least squares EQ
         rx_sym_pilots = self.do_pilot_eq_one(num_modem_frames,rx_sym_pilots)
         rx_sym = rx_sym_pilots[:,:,1:self.Ns+1,:]
         rx_sym = torch.reshape(rx_sym, (1, -1, self.latent_dim//2))
         z_hat = torch.zeros(1,rx_sym.shape[1], self.latent_dim)

         z_hat[:,:,::2] = rx_sym.real
         z_hat[:,:,1::2] = rx_sym.imag
      else:
         # Simpler (but lower performance) EQ as average of pilots, as LS set up for PDDDDP, rather than out PEDDDE
         for c in range(self.Nc):
            phase_offset = torch.angle(rx_sym_pilots[0,0,0,c]/self.P[c] +
                           rx_sym_pilots[0,0,1,c]/self.Pend[c] +
                           rx_sym_pilots[0,0,Ns,c]/self.Pend[c])
            rx_sym_pilots[:,:,:Ns+1,c] *= torch.exp(-1j*phase_offset)
         rx_sym = torch.reshape(rx_sym_pilots[:,:,2:Ns,:],(1,(Ns-2)*self.Nc))
         z_hat = torch.zeros(1,(Ns-2)*self.Nc*2)

         z_hat[:,::2] = rx_sym.real
         z_hat[:,1::2] = rx_sym.imag
 
      return z_hat


# Generate root raised cosine (Root Nyquist) filter coefficients
# thanks http://www.dsplog.com/db-install/wp-content/uploads/2008/05/raised_cosine_filter.m

def gen_rn_coeffs(alpha, T, Rs, Nsym, M):

  Ts = 1/Rs

  n = np.arange(-Nsym*Ts/2,Nsym*Ts/2,T)
  Nfilter = Nsym*M
  Nfiltertiming = M+Nfilter+M

  sincNum = np.sin(np.pi*n/Ts) # numerator of the sinc function
  sincDen = np.pi*n/Ts    # denominator of the sinc function
  sincDenZero = np.argwhere(np.abs(sincDen) < 10**-10)
  sincOp = sincNum/sincDen
  sincOp[sincDenZero] = 1; # sin(pix)/(pix) = 1 for x=0

  cosNum = np.cos(alpha*np.pi*n/Ts)
  cosDen = 1-(2*alpha*n/Ts)**2
  cosDenZero = np.argwhere(np.abs(cosDen)<10**-10)
  cosOp = cosNum/cosDen
  cosOp[cosDenZero] = np.pi/4
  gt_alpha5 = sincOp*cosOp
  Nfft = 4096
  GF_alpha5 = np.fft.fft(gt_alpha5,Nfft)/M

  # sqrt causes stop band to be amplified, this hack pushes it down again
  for i in np.arange(0,Nfft):
    if np.abs(GF_alpha5[i]) < 0.02:
      GF_alpha5[i] *= 0.001

  GF_alpha5_root = np.sqrt(np.abs(GF_alpha5)) * np.exp(1j*np.angle(GF_alpha5))
  ifft_GF_alpha5_root = np.fft.ifft(GF_alpha5_root)
  return ifft_GF_alpha5_root[np.arange(0,Nfilter)].real

def sample_clock_offset(tx, sample_clock_offset_ppm):
   tin=0
   tout=0
   rx = np.zeros(len(tx), dtype=np.csingle)
   while tin+1 < len(tx) and tout < len(rx):
      t1 = int(np.floor(tin))
      t2 = int(np.ceil(tin))
      f = tin - t1
      rx[tout] = (1-f)*tx[t1] + f*tx[t2]
      tout += 1
      tin  += 1 + sample_clock_offset_ppm/1E6
   return rx

# single carrier PSK modem, suitable for baseband FM channel (DC coupled or band pass), 
# or directly over a VHF/UHF
class single_carrier:
   def __init__(self, Rs=2400, Fs=9600, fcentreHz=0, alpha=0.25):
      self.fcentreHz = fcentreHz
      self.alpha = alpha
      self.Fs = Fs
      self.T = 1/self.Fs
      self.Rs = Rs
      self.Nfilt_sym = 6
      self.M = int(self.Fs/self.Rs)
      # M must be an integer
      assert self.M == self.Fs/self.Rs
      self.lo_omega_rect = np.exp(1j*2*np.pi*fcentreHz/self.Fs)
      
      # +++++-++--++----+-+-----
      self.p25_frame_sync = np.array([1,1,1,1,1,-1,1,1,-1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,-1], dtype=np.csingle)
      self.Nsync_syms = 16
      self.Nframe_syms = 96
      self.Npayload_syms = self.Nframe_syms - self.Nsync_syms
      p = self.p25_frame_sync[:self.Nsync_syms]
      self.p_scale = np.dot(p,p)/np.sqrt(np.dot(p,p))
      self.sync_thresh = 0.5
      self.unsync_thresh1 = 2
      self.unsync_thresh2 = 3

      self.rrc = gen_rn_coeffs(self.alpha, self.T, self.Rs, self.Nfilt_sym, self.M)
      self.Ntap = len(self.rrc)
      self.tx_filt_mem = np.zeros(self.Ntap, dtype=np.csingle)
      self.rx_filt_mem = np.zeros(self.Ntap, dtype=np.csingle)
      self.rx_filt_out = np.zeros((self.Nframe_syms+2)*self.M, dtype=np.csingle)

      self.sample_point = 5
      self.nin = self.Nframe_syms*self.M
      self.rx_symb_buf = np.zeros(2*self.Nframe_syms, dtype=np.csingle)

      self.Nphase = 21
      # Nphase must be odd
      assert np.mod(self.Nphase,2) == 1
      self.phase_est_fine = 0
      self.phase_est_coarse = 0
      self.phase_est_mem = np.zeros(self.Nphase, dtype=np.csingle)
      self.phase_est_log = np.zeros(self.Nframe_syms, dtype=np.csingle)
      self.phase_ambiguity = 0

      #self.phase_est_log = np.array([], dtype=np.csingle)
      rx_symb_buf = np.zeros(2*self.Nframe_syms, dtype=np.csingle)

      self.tx_lo_phase_rect = 1 + 1j*0
      self.rx_lo_phase_rect = 1 + 1j*0
      
      self.state = "search"
      self.fs_s = 0
      self.g = 1

      # 4x oversampling filter for timing offset simulation
      self.lpf = complex_bpf(101,self.Fs*4,self.Fs, 0)
      # create a RNG with same sequence for BER testing with separate tx and rx
      seed = 65647437836358831880808032086803839626
      self.rng = np.random.default_rng(seed)

   # input rate Rs symbols, output at rate Fs, preserves memory for next call
   def tx(self, tx_symbs):
      assert len(tx_symbs) == 80

      # pre-pend frame sync word
      tx_symbs = np.concatenate([self.p25_frame_sync[:self.Nsync_syms],tx_symbs])
 
      # RN filter
      tx_filt_in = np.concatenate([self.tx_filt_mem, np.zeros(len(tx_symbs)*self.M, dtype=np.csingle)])
      tx_filt_in[self.Ntap::self.M] = tx_symbs*self.M
      tx_filt_out = np.zeros(len(tx_symbs)*self.M, dtype=np.csingle)
      for i in np.arange(0,len(tx_symbs)*self.M):
         tx_filt_out[i] = np.dot(tx_filt_in[i+1:i+self.Ntap+1],self.rrc)
      self.tx_filt_mem = tx_filt_in[-self.Ntap:]

      # freq shift up to centre freq
      for i in np.arange(len(tx_filt_out)):
         tx_filt_out[i] *= self.tx_lo_phase_rect
         self.tx_lo_phase_rect *= self.lo_omega_rect
         #print(self.tx_lo_phase_rect)
      # normalise tx LO rect cooordinates  
      self.tx_lo_phase_rect /= np.abs(self.tx_lo_phase_rect)
 
      return tx_filt_out

   # estimate fine timing and resample (decimate) at optimum timing estimate, returning 
   # rate Rs symbols for this frame
   def est_timing_and_decimate(self, rx_filt):
      # rx_filt contains (1+Nframe_syms+1)*M samples.  The current frame being demodulated is the 
      # middle Nframe_syms. The extra samples at the start and end are to cope with fine timing
      # requiring samples just outside the current frame 
      M = self.M

      # Fine Timing:
      #
      # The envelope has a frequency component at the symbol rate.  The
      # phase of this frequency component indicates the optimum sampling
      # point (modulo one symbol). We compute the phase with a single point
      # DFT at frequency 2*pi/M

      # we estimate fine timing referenced to the nominal sample_point
      env = np.abs(rx_filt[int(self.sample_point):])
      x = np.dot(env,np.exp(-1j*2*np.pi*np.arange(0,len(env))/M))
      norm_rx_timing = np.angle(x)/(2*np.pi) # -0.5 ... 0.5
      rx_timing = norm_rx_timing*M           # -M/2 ... M/2

      # correct sampling instant in opp direction to timing offset
      rx_timing_correction = -rx_timing   
      
      # Use linear interpolation to resample at the optimum sampling point.  We assume fine timing
      # is a constant over the frame
      low_sample = int(np.floor(rx_timing_correction))
      fract = rx_timing_correction - low_sample
      sample = self.sample_point + low_sample + np.arange(0,self.Nframe_syms*M,M)
      rx_symbols = rx_filt[sample]*(1-fract) + rx_filt[sample+1]*fract

      # adjust number of samples for next frame to keep timing in the sweet spot
      self.nin = self.Nframe_syms*M
      if norm_rx_timing < -0.35:
         self.nin += M/4
      if norm_rx_timing > 0.35:
         self.nin -= M/4
      self.nin = int(self.nin)

      self.norm_rx_timing = norm_rx_timing
      
      return rx_symbols

   # estimate the phase offset and return phase corrected symbol
   def est_phase_and_correct(self,rx_symbs):
      mod_order = 2

      # maintain a buffer of symbols, current sample is at centre of self.Nphase sample window
      symbol_buf = np.concatenate([self.phase_est_mem, rx_symbs])
   
      rx_symbs_corrected = np.zeros(len(rx_symbs), dtype=np.csingle)
      for s in np.arange(0,len(rx_symbs)):
         # strip (BPSK) modulation by taking symbol to mod_order power, note this means estimate is 
         # modulo 2*pi/mod_order.  We track jumps in phase due to this modulo effect, and the initial 
         # ambiguity is resolved with the frame sync word.

         # sum a window of modulation stripped symbols to get a good average centred on this symbol
         phase_est_fine = np.angle(sum((symbol_buf[s+1:s+1+self.Nphase])**mod_order))/mod_order

         # track phase jummps and adjust coarse past of phase est
         if phase_est_fine - self.phase_est_fine < -0.9*np.pi:
            self.phase_est_coarse += np.pi
         if phase_est_fine - self.phase_est_fine > 0.9*np.pi:
            self.phase_est_coarse -= np.pi
         self.phase_est_fine = phase_est_fine
         phase_est = self.phase_est_coarse + self.phase_est_fine

         # update log for plotting purposes
         self.phase_est_log[s] = np.exp(1j*phase_est)
         
         # correct phase of symbol at the centre of window
         centre = s + self.Nphase//2
         rx_symbs_corrected[s] = symbol_buf[centre]*np.exp(-1j*phase_est)

      self.phase_est_mem = symbol_buf[-self.Nphase:]

      return rx_symbs_corrected
   
   # rx a single frame's worth of samples, output rate Rs symbols, performs filtering, phase and timing correction
   # input rate Fs, output rate Rs, preserves memory for next call
   def rx_Fs_to_Rs(self, rx_samples):
      assert len(rx_samples) == self.nin
      M = self.M
      
      # shift samples from centre freq to baseband
      rx_bb_samples = np.copy(rx_samples)
      for i in np.arange(len(rx_samples)):
         rx_bb_samples[i] *= self.rx_lo_phase_rect
         self.rx_lo_phase_rect *= np.conj(self.lo_omega_rect)
      # normaliase rx LO rect cooordinates  
      self.rx_lo_phase_rect /= np.abs(self.rx_lo_phase_rect)

      # filter received sample stream
      rx_filt_in = np.concatenate([self.rx_filt_mem, rx_bb_samples])
      to_keep = len(self.rx_filt_out) - self.nin
      self.rx_filt_out[:to_keep] = self.rx_filt_out[-to_keep:]
      for i in np.arange(0,self.nin):
         self.rx_filt_out[to_keep+i] = np.dot(rx_filt_in[i+1:i+self.Ntap+1],self.rrc)
      self.rx_filt_mem = rx_filt_in[-self.Ntap:]

      rx_symbs = self.est_timing_and_decimate(self.rx_filt_out)
      rx_symbs = self.est_phase_and_correct(rx_symbs)

      return rx_symbs
   
   # rx nin samples, outputs demodulated and frame aligned symbols, handles frame sync
   def rx(self, rx_samples):
      assert len(rx_samples) == self.nin
      Nframe_syms = self.Nframe_syms
      Nsync_syms = self.Nsync_syms
      fs_s = self.fs_s

      # demod next frame of symbols
      self.rx_symb_buf[:Nframe_syms] = self.rx_symb_buf[Nframe_syms:] 
      self.rx_symb_buf[Nframe_syms:] = self.rx_Fs_to_Rs(rx_samples)

      # iterate sync state machine ----------------------------------------------

      next_state = self.state
      if self.state == "search":
         # look for frame sync in two frame buffer, Cs is a normalised correlation at
         # symbol s (ref freedv_low.pdf "Coarse Timing Estimation").  Note we perform
         # a floating point cross-correlation rather than just error count in FS word,
         # as we use the sign of the correlation to resolve the phase ambiguity 
         max_Cs = 0
         max_s = 0
         for s in np.arange(0,Nframe_syms):
            rx_symbs = self.rx_symb_buf[s:s+Nsync_syms]
            num = np.dot(np.conj(rx_symbs),self.p25_frame_sync[:self.Nsync_syms]/self.p_scale)
            denom = np.sqrt(np.dot(np.conj(rx_symbs),rx_symbs))
            Cs = num/(denom+1E-12)
            if np.abs(Cs) > np.abs(max_Cs):
               max_s = s
               max_Cs = Cs
         self.max_Cs = max_Cs

         if np.abs(max_Cs) >= self.sync_thresh:
            next_state = "sync"
            fs_s = max_s
            self.fs_s = fs_s
            self.bad_fs = 0

            # resolve phase ambiguity based on frame sync, we assume phase est tracks from here
            if max_Cs.real < 0:
               self.phase_ambiguity = np.pi
            else:
               self.phase_ambiguity = 0
         
            # amplitude norm based on frame sync word
            fs_symbs = self.rx_symb_buf[fs_s:fs_s+Nsync_syms]
            self.g = 1/(np.mean(np.abs(fs_symbs)**2)**0.5+1E-12)

      if self.state == "sync":
         # count errors in FS word to see if we are still in sync
         rx_symbs = np.exp(1j*self.phase_ambiguity)*self.rx_symb_buf[fs_s:fs_s+Nsync_syms]
         n_errors = np.sum(rx_symbs * self.p25_frame_sync[:self.Nsync_syms] < 0)
         if n_errors > self.unsync_thresh1:
            self.bad_fs += 1
         else:
            self.bad_fs = 0
         if self.bad_fs >= self.unsync_thresh2:
            next_state = "search"

         # amplitude norm based on frame sync word
         fs_symbs = self.rx_symb_buf[fs_s:fs_s+Nsync_syms]
         self.g = 1/(np.mean(np.abs(fs_symbs)**2)**0.5+1E-12)

      self.state = next_state

      # return payload symbols
      return np.exp(1j*self.phase_ambiguity)*self.rx_symb_buf[fs_s+Nsync_syms:fs_s+Nframe_syms]


   # python3 -c "from radae import single_carrier; s=single_carrier(); s.run_test(100,sample_clock_offset_ppm=-100,plots_en=True)"
   def run_test(self,Nframes=10, EbNodB=100, phase_off=0, freq_off=0, mag=1, sample_clock_offset_ppm=0, target_ber=0, plots_en=False):
      Nframe_syms = self.Nframe_syms
      Npayload_syms = self.Npayload_syms

      # single fixed test frame
      tx_symbs = 1 - 2*(self.rng.random(Npayload_syms) > 0.5) + 0*1j

      # create a stream of tx samples
      tx = np.array([], dtype=np.csingle)
      for f in np.arange(0,Nframes):
         atx = self.tx(tx_symbs)
         tx = np.concatenate([tx,atx])
      
      # simulate timing offset, 4x oversampling then linear interpolation
      # TODO: is ppm correct when we oversample by 4 ?
      tx_zp = np.zeros(4*len(tx), dtype=np.csingle)
      tx_zp[0::4] = tx
      tx_4 = self.lpf.bpf(tx_zp)
      rx = sample_clock_offset(tx_4, sample_clock_offset_ppm)[0::4]

      # simulate freq, phase, mag offsets and AWGN noise
      phase_vec = 2*np.pi*freq_off*np.arange(0,len(rx))/self.Fs + phase_off
      rx *= np.exp(1j*phase_vec)
      sigma = np.sqrt(1/(self.M*10**(EbNodB/10)))
      noise = (sigma/np.sqrt(2))*(self.rng.standard_normal(len(rx)) + 1j*self.rng.standard_normal(len(rx)))
      rx = mag*(rx + noise)

      # demodulate stream with rx
      rx_symb_log = np.array([], dtype=np.csingle)
      error_log = np.array([])
      norm_rx_timing_log = np.array([])
      phase_est_log = np.array([], dtype=np.csingle)
      nin = self.nin
      total_errors = 0
      total_bits = 0

      n = 0
      while len(rx[n:]) >= nin:
         # demod next frame
         rx_symbs = self.rx(rx[n:n+nin])
         print(f"state: {self.state:6} nin: {self.nin:4d} rx_timing: {self.norm_rx_timing:5.2f} max_metric: {self.max_Cs.real:5.2f}", end='')
         if self.state == "sync":
            n_errors = np.sum(rx_symbs * tx_symbs < 0)
            error_log = np.append(error_log,n_errors)
            total_errors += n_errors
            total_bits += len(tx_symbs)
            print(f" fs_s: {self.fs_s:4d} ph_ambig: {self.phase_ambiguity:5.2f} g: {self.g:5.2f} n_errors: {n_errors:4d}", end='')

            # update logs for plotting
            rx_symb_log = np.concatenate([rx_symb_log,self.rx_symb_buf[Nframe_syms:]])
            norm_rx_timing_log = np.append(norm_rx_timing_log, self.norm_rx_timing)
            phase_est_log = np.append(phase_est_log, self.phase_est_log)
         n += nin
         nin = self.nin
         print()

      ber = 0
      if total_bits:
         ber = total_errors/total_bits
      print(f"total_bits: {total_bits:4d} total_errors: {total_errors:4d} BER: {ber:5.4f} Target BER: {target_ber:5.4f}")

      if plots_en:
         plt.figure(1)
         plt.plot(self.g*rx_symb_log.real,self.g*rx_symb_log.imag,'+'); plt.ylabel('Symbols')
         plt.axis([-2,2,-2,2])
         plt.figure(2)
         plt.subplot(211)
         plt.plot(error_log); plt.ylabel('Errors/frame')
         plt.subplot(212)
         plt.plot(norm_rx_timing_log,'+'); plt.ylabel('Fine Timing')
         plt.axis([0,len(norm_rx_timing_log),-0.5,0.5])
         plt.figure(3)
         plt.plot(np.angle(phase_est_log),'+'); plt.title('Phase Est')
         plt.axis([0,len(phase_est_log),-np.pi,np.pi])
         plt.figure(4)
         from matplotlib.mlab import psd
         P, frequencies = psd(rx,Fs=self.Fs)
         PdB = 10*np.log10(P)
         plt.plot(frequencies,PdB); plt.title('PSD'); plt.grid()
         mx = 10*np.ceil(max(PdB)/10)
         plt.axis([-self.Fs/2, self.Fs/2, mx-40, mx])
         plt.show()

      test_pass = ber <= target_ber
      if test_pass:
         print("PASS")
      else:
         print("FAIL")
      return test_pass
   
   # TODO 
   # * DC offset removal
   # * separate tx/rx cmd line applications that talk to BBFM ML enc/dec
 
# python3 -c "from radae import single_carrier,single_carrier_tests; single_carrier_tests()"
def single_carrier_tests():
      modem = single_carrier()
      total = 0; passes = 0

      # baseline test with vanilla channel
      total += 1; modem = single_carrier(); passes += modem.run_test()

      # sample clock offsets
      total += 1; modem = single_carrier(); passes += modem.run_test(Nframes=100, sample_clock_offset_ppm=100)
      total += 1; modem = single_carrier(); passes += modem.run_test(Nframes=100, sample_clock_offset_ppm=-100)

      # BER test: allow 0.5dB implementation loss
      EbNodB = 4
      target_ber = 0.5*math.erfc(np.sqrt(10**((EbNodB-0.5)/10)))

      # DC coupled 
      total += 1; modem = single_carrier(); passes += modem.run_test(Nframes=100, sample_clock_offset_ppm=-100, EbNodB=EbNodB, target_ber=target_ber)

      # 1500 Hz centre freq     
      total += 1; modem = single_carrier(fcentreHz=1500); passes += modem.run_test(Nframes=100, 
                                                                                   sample_clock_offset_ppm=-100, 
                                                                                   EbNodB=EbNodB, 
                                                                                   freq_off=1,
                                                                                   mag=100,
                                                                                   target_ber=target_ber)

      print(f"{passes:d}/{total:d}")

      if passes == total:
         print("ALL PASS")
