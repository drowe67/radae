"""

  Radio Autoencoder streaming receiver, "embeddded" version.
  
  rate Fs complex float samples in, features out.
  rate Fs real int16 samples in, features out.

  Designed to connected to a SDR to perform real time RADAE decoding on 
  received sample streams.  Full function state machine and continous 
  updates of timing, freq offsets and amplituide estimates.
  
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

import os, sys, struct
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf,acquisition,receiver_one

# Hard code all this for now to avoid arg passing complexities TODO: consider a way to pass in at init-time
latent_dim = 80
auxdata = True
bottleneck = 3
use_stdout=True
bpf=True
v=2

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

nb_total_features = 36
num_used_features = 20
num_features = 20
if auxdata:
    num_features += 1

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, 
              pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
              coarse_mag=True,time_offset=-16, bottleneck=bottleneck)
model.eval()

# check a bunch of model options we rely on for receiver to work
assert model.pilots and model.pilot_eq
assert model.per_carrier_eq
assert model.eq_mean6 == False   # we are using least squares algorithm
assert model.phase_mag_eq == False
assert model.coarse_mag
receiver = receiver_one(model.latent_dim,model.Fs,model.M,model.Ncp,model.Wfwd,model.Nc,
                        model.Ns,model.w,model.P,model.bottleneck,model.pilot_gain,
                        model.time_offset,model.coarse_mag)

M = model.M
Ncp = model.Ncp
Ns = model.Ns               # number of data symbols between pilots
Nmf = int((Ns+1)*(M+Ncp))   # number of samples in one modem frame
Nc = model.Nc
p = np.array(model.p) 
Fs = model.Fs
Rs = model.Rs
w = np.array(model.w)

if bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
   bpf = complex_bpf(Ntap, model.Fs, bandwidth,centre)

acq = acquisition(Fs,Rs,M,Ncp,Nmf,p,model.pend)

Tunsync = 3.0                        # allow some time before lossing sync to ride over fades
Nmf_unsync = int(Tunsync*Fs/Nmf)
endofover = False
uw_error_thresh = 7 # P(reject|correct) = 1 -  binocdf(8,24,0.1) = 4.5E-4
                    # P(accept|false)   = binocdf(8,24,0.5)      = 3.2E-3
synced_count_one_sec = Fs//Nmf

# P DDD P DDD P Ncp
# extra Ncp at end so we can handle timing slips
rx_buf = np.zeros(2*Nmf+M+Ncp,np.csingle)
rx = np.zeros(0,np.csingle)
z_hat_log = torch.zeros(0,model.Nzmf,model.latent_dim)

class radae_rx:
   def __init__(self,model_name):
      self.nin = Nmf
      self.state = "search"
      self.tmax_candidate = 0
      self.mf = 1
      self.valid_count = 0
      self.uw_errors = 0
      self.synced_count = 0
      self.rx_phase = 1 + 1j*0

      checkpoint = torch.load(model_name, map_location='cpu',weights_only=True)
      model.load_state_dict(checkpoint['state_dict'], strict=False)
      # Stateful decoder wasn't present during training, so we need to load weights from existing decoder
      model.core_decoder_statefull_load_state_dict()

   def get_n_floats_out(self):
         return model.Nzmf*model.dec_stride*nb_total_features
                 
   def get_nin_max(self):
         return Nmf+M
   
   def get_nin(self):
         return self.nin
                 
   def do_radae_rx(self, buffer_complex, features_out):
      with torch.inference_mode():
         prev_state = self.state
         valid_output = False
         endofover = False
         uw_fail = False
         buffer_complex = buffer_complex[:self.nin]
         if bpf:
            buffer_complex = bpf.bpf(buffer_complex)
         rx_buf[:-self.nin] = rx_buf[self.nin:]                           # out with the old
         rx_buf[-self.nin:] = buffer_complex                         # in with the new
         if self.state == "search" or self.state == "candidate":
            candidate, self.tmax, self.fmax = acq.detect_pilots(rx_buf)
         else:
            # we're in sync, so check we can still see pilots and run receiver
            ffine_range = np.arange(self.fmax-1,self.fmax+1,0.1)
            tfine_range = np.arange(max(0,self.tmax-8),self.tmax+8)
            self.tmax,fmax_hat = acq.refine(rx_buf, self.tmax, self.fmax, tfine_range, ffine_range)
            self.fmax = 0.9*self.fmax + 0.1*fmax_hat
            candidate,endofover = acq.check_pilots(rx_buf,self.tmax,self.fmax)

            # handle timing slip when rx sample clock > tx sample clock
            self.nin = Nmf
            if self.tmax >= Nmf-M:
               self.nin = Nmf + M
               self.tmax -= M
               #print("slip+", file=sys.stderr)
            # handle timing slip when rx sample clock < tx sample clock
            if self.tmax < M:
               self.nin = Nmf - M
               self.tmax += M
               #print("slip-", file=sys.stderr)

            self.synced_count += 1
            if self.synced_count % synced_count_one_sec == 0:
               if self.uw_errors > uw_error_thresh:
                  uw_fail = True
               self.uw_errors = 0

            if not endofover:
               # correct frequency offset, note we preserve state of phase
               # TODO do we need preserve state of phase?  We're passing entire vector and there isn't any memory (I think)
               w = 2*np.pi*self.fmax/Fs
               rx_phase_vec = np.zeros(Nmf+M+Ncp,np.csingle)
               for n in range(Nmf+M+Ncp):
                  self.rx_phase = self.rx_phase*np.exp(-1j*w)
                  rx_phase_vec[n] = self.rx_phase
               rx1 = rx_buf[self.tmax-Ncp:self.tmax-Ncp+Nmf+M+Ncp]
               rx = torch.tensor(rx1*rx_phase_vec, dtype=torch.complex64)

               # run through RADAE receiver DSP
               z_hat = receiver.receiver_one(rx)
               # decode z_hat to features
               assert(z_hat.shape[1] == model.Nzmf)
               features_hat = model.core_decoder_statefull(z_hat)
               if auxdata:
                  symb_repeat = 4
                  aux_symb = features_hat[:,:,20].detach().numpy()
                  aux_bits = 1*(aux_symb[0,::symb_repeat] > 0)
                  features_hat = features_hat[:,:,0:20]
                  self.uw_errors += np.sum(aux_bits)

               # add unused features and output
               features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
               features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
               np.copyto(features_out, features_hat)
               valid_output = True
            
         if v == 2 or (v == 1 and (self.state == "search" or self.state == "candidate" or prev_state == "candidate")):
            print(f"{self.mf:3d} state: {self.state:10s} valid: {candidate:d} {endofover:d} {self.valid_count:2d} Dthresh: {acq.Dthresh:8.2f} ", end='', file=sys.stderr)
            print(f"Dtmax12: {acq.Dtmax12:8.2f} {acq.Dtmax12_eoo:8.2f} tmax: {self.tmax:4d} fmax: {self.fmax:6.2f}", end='', file=sys.stderr)
            if auxdata and self.state == "sync":
               print(f" aux: {aux_bits:} uw_err: {self.uw_errors:d}", file=sys.stderr)
            else:
               print("",file=sys.stderr)

         # iterate state machine  
         next_state = self.state
         prev_state = self.state
         if self.state == "search":
            if candidate:
               next_state = "candidate"
               self.tmax_candidate = self.tmax
               self.valid_count = 1
         elif self.state == "candidate":
            # look for 3 consecutive matches with about the same timing offset  
            if candidate and np.abs(self.tmax-self.tmax_candidate) < 0.02*M:
               self.valid_count = self.valid_count + 1
               if self.valid_count > 3:
                  next_state = "sync"
                  self.synced_count = 0
                  uw_fail = False
                  if auxdata:
                     self.uw_errors = 0
                  self.valid_count = Nmf_unsync
                  ffine_range = np.arange(self.fmax-10,self.fmax+10,0.25)
                  tfine_range = np.arange(self.tmax-1,self.tmax+2)
                  self.tmax,self.fmax = acq.refine(rx_buf, self.tmax, self.fmax, tfine_range, ffine_range)
            else:
               next_state = "search"
         elif self.state == "sync":
            # during some tests it's useful to disable these unsync features
            unsync_enable = True

            if candidate:
               self.valid_count = Nmf_unsync
            else:
               self.valid_count -= 1
               if unsync_enable and self.valid_count == 0:
                  next_state = "search"

            if unsync_enable and (endofover or uw_fail):
               next_state = "search"

         self.state = next_state
         self.mf += 1

         return valid_output

if __name__ == '__main__':
   rx = radae_rx(model_name = "../model19_check3/checkpoints/checkpoint_epoch_100.pth")

   # allocate storage for output features
   features_out = np.zeros(rx.get_n_floats_out(),dtype=np.float32)
   while True:
      buffer = sys.stdin.buffer.read(rx.get_nin()*struct.calcsize("ff"))
      if len(buffer) != rx.get_nin()*struct.calcsize("ff"):
         break
      buffer_complex = np.frombuffer(buffer,np.csingle)
      valid_output = rx.do_radae_rx(buffer_complex, features_out)
      if valid_output:
         sys.stdout.buffer.write(features_out)