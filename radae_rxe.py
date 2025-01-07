"""

  Radio Autoencoder streaming receiver, "embeddded" version.
  
  rate Fs complex float samples in, features out.

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

import os, sys, struct,argparse
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf,acquisition,receiver_one

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

nb_total_features = 36
num_used_features = 20

Tunsync = 3.0                        # allow some time before lossing sync to ride over fades
uw_error_thresh = 7 # P(reject|correct) = 1 -  binocdf(8,24,0.1) = 4.5E-4
                    # P(accept|false)   = binocdf(8,24,0.5)      = 3.2E-3

class radae_rx:
   def __init__(self, model_name, latent_dim = 80, auxdata = True, bottleneck = 3, bpf_en=True, v=2, disable_unsync=False, foff_err=0, bypass_dec=False):

      self.latent_dim = latent_dim
      self.auxdata = auxdata
      self.bottleneck = bottleneck
      self.bpf_en = bpf_en
      self.v = v
      self.disable_unsync = disable_unsync
      self.foff_err = foff_err
      self.bypass_dec = bypass_dec

      print(f"bypass_dec: {bypass_dec} foff_err: {foff_err:f}", file=sys.stderr)

      self.num_features = 20
      if self.auxdata:
         self.num_features += 1

      self.model = RADAE(self.num_features, latent_dim, EbNodB=100, rate_Fs=True, 
                        pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
                        coarse_mag=True,time_offset=-16, bottleneck=bottleneck)
      model = self.model
      self.model.eval()

      # check a bunch of model options we rely on for receiver to work
      assert self.model.pilots and model.pilot_eq
      assert self.model.per_carrier_eq
      assert self.model.eq_mean6 == False   # we are using least squares algorithm
      assert self.model.phase_mag_eq == False
      assert self.model.coarse_mag
   
      self.receiver = receiver_one(model.latent_dim,model.Fs,model.M,model.Ncp,model.Wfwd,model.Nc,
                              model.Ns,model.w,model.P,model.bottleneck,model.pilot_gain,
                              model.time_offset,model.coarse_mag)

      M = model.M
      Ncp = model.Ncp
      Ns = model.Ns               # number of data symbols between pilots
      Nmf = int((Ns+1)*(M+Ncp))   # number of samples in one modem frame
      self.Nmf = Nmf
      Nc = model.Nc
      p = np.array(model.p) 
      Fs = model.Fs
      Rs = model.Rs
      w = np.array(model.w)

      if bpf_en:
         Ntap=101
         bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
         centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
         print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
         self.bpf = complex_bpf(Ntap, model.Fs, bandwidth,centre)

      self.acq = acquisition(Fs,Rs,M,Ncp,Nmf,p,model.pend)

      if not self.bypass_dec:
         # load model from a checkpoint file
         checkpoint = torch.load(model_name, map_location='cpu',weights_only=True)
         model.load_state_dict(checkpoint['state_dict'], strict=False)
         # Stateful decoder wasn't present during training, so we need to load weights from existing decoder
         model.core_decoder_statefull_load_state_dict()

      # number of input floats per processing frame
      if not self.bypass_dec:
         self.n_floats_out = model.Nzmf*model.enc_stride*nb_total_features
      else:
         self.n_floats_out = model.Nzmf*self.latent_dim

      self.Nmf_unsync = int(Tunsync*Fs/Nmf)
 
      self.nin = Nmf
      self.state = "search"
      self.tmax_candidate = 0
      self.mf = 1
      self.valid_count = 0
      self.uw_errors = 0
      self.synced_count = 0
      self.rx_phase = 1 + 1j*0

      self.synced_count_one_sec = Fs//Nmf

      # P DDD P DDD P Ncp
      # extra Ncp at end so we can handle timing slips
      self.rx_buf = np.zeros(2*Nmf+M+Ncp,np.csingle)
      self.z_hat_log = torch.zeros(0,model.Nzmf,model.latent_dim)

   def get_n_features_out(self):
      return self.model.Nzmf*self.model.dec_stride*nb_total_features
                 
   def get_n_floats_out(self):
      return self.n_floats_out
                 
   def get_nin_max(self):
      return self.Nmf+self.model.M
   
   def get_nin(self):
      return self.nin
                 
   def get_sync(self):
      return self.state == "sync"

   def get_snrdB_3k_est(self):
      return int(self.receiver.snrdB_3k_est)

   def sum_uw_errors(self,new_uw_errors):
      self.uw_errors += new_uw_errors

   def do_radae_rx(self, buffer_complex, floats_out):
      acq = self.acq
      bpf = self.bpf
      receiver = self.receiver
      model = self.model
      M = model.M
      Ncp = model.Ncp
      Ns = model.Ns               
      Nmf = int((Ns+1)*(M+Ncp)) 
      Fs = model.Fs
      w = np.array(model.w)
      Nmf = self.Nmf
      auxdata = self.auxdata
      v = self.v
      rx_buf = self.rx_buf
      aux_bits = np.zeros(model.Nzmf,dtype=np.int16)

      with torch.inference_mode():
         prev_state = self.state
         valid_output = False
         endofover = False
         uw_fail = False
         buffer_complex = buffer_complex[:self.nin]
         if self.bpf_en:
            buffer_complex = bpf.bpf(buffer_complex)
         rx_buf[:-self.nin] = rx_buf[self.nin:]                      # out with the old
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
            if self.synced_count % self.synced_count_one_sec == 0:
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
               valid_output = True
            
         if v == 2 or (v == 1 and (self.state == "search" or self.state == "candidate" or prev_state == "candidate")):
            print(f"{self.mf:3d} state: {self.state:10s} valid: {candidate:d} {endofover:d} {self.valid_count:2d} Dthresh: {acq.Dthresh:8.2f} ", end='', file=sys.stderr)
            print(f"Dtmax12: {acq.Dtmax12:8.2f} {acq.Dtmax12_eoo:8.2f} tmax: {self.tmax:4d} fmax: {self.fmax:6.2f}", end='', file=sys.stderr)
            print(f" SNRdB: {receiver.snrdB_3k_est:5.2f}", end='', file=sys.stderr)
            if auxdata and self.state == "sync":
               print(f" uw_err: {self.uw_errors:d}", file=sys.stderr)
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
            if candidate and np.abs(self.tmax-self.tmax_candidate) < Ncp:
               self.valid_count = self.valid_count + 1
               if self.valid_count > 3:
                  next_state = "sync"
                  model.core_decoder_statefull.module.reset()
                  self.synced_count = 0
                  uw_fail = False
                  if auxdata:
                     self.uw_errors = 0
                  self.valid_count = self.Nmf_unsync
                  ffine_range = np.arange(self.fmax-10,self.fmax+10,0.25)
                  tfine_range = np.arange(max(0,self.tmax-1),self.tmax+2)
                  self.tmax,self.fmax = acq.refine(rx_buf, self.tmax, self.fmax, tfine_range, ffine_range)
                  # testing: only insert freq offset error on first sync
                  self.fmax += self.foff_err
                  self.foff_err = 0
            else:
               next_state = "search"
         elif self.state == "sync":
            # during some tests it's useful to disable these unsync features
            unsync_enable = True
            if self.disable_unsync:
               if self.synced_count > int(self.disable_unsync*Fs/Nmf):
                  unsync_enable = False

            if candidate:
               self.valid_count = self.Nmf_unsync
            else:
               self.valid_count -= 1
               if unsync_enable and self.valid_count == 0:
                  next_state = "search"

            if unsync_enable and (endofover or uw_fail):
               next_state = "search"

         self.state = next_state
         self.mf += 1

         # We call core decoder at end to model behaivour with external C core decoder
         if valid_output:
            assert(z_hat.shape[1] == model.Nzmf)
            if not self.bypass_dec:
               # decode z_hat to features
               features_hat = model.core_decoder_statefull(z_hat)
               if auxdata:
                  symb_repeat = 4
                  aux_symb = features_hat[:,:,20].detach().numpy()
                  aux_bits = 1*(aux_symb[0,::symb_repeat] > 0)
                  #print(aux_symb,aux_symb[0,::symb_repeat],aux_bits,file=sys.stderr)

                  features_hat = features_hat[:,:,0:20]
                  self.sum_uw_errors(np.sum(aux_bits))

               # add unused features and output
               features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
               features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
               np.copyto(floats_out, features_hat)
            else:
               np.copyto(floats_out, z_hat.cpu().detach().numpy().flatten().astype('float32'))

         return valid_output

if __name__ == '__main__':
   parser = argparse.ArgumentParser(description='RADAE streaming receiver, IQ.f32 on stdin to features.f32 on stdout')
   parser.add_argument('--model_name', type=str, help='path to model in .pth format', default="model19_check3/checkpoints/checkpoint_epoch_100.pth")
   parser.add_argument('--noauxdata', dest="auxdata", action='store_false', help='disable injectiopn of auxillary data symbols')
   parser.add_argument('-v', type=int, default=2, help='Verbose level (default 2)')
   parser.add_argument('--disable_unsync', type=float, default=0.0, help='test mode: disable auxdata based unsyncs after this many seconds (default disabled)')
   parser.add_argument('--no_stdout', action='store_false', dest='use_stdout', help='disable the use of stdout (e.g. with python3 -m cProfile)')
   parser.add_argument('--foff_err', type=float, default=0.0, help='Artifical freq offset error after first sync to test false sync (default 0.0)')
   parser.add_argument('--bypass_dec', action='store_true', help='Bypass core decoder, write z_hat to stdout')
   parser.set_defaults(auxdata=True)
   parser.set_defaults(use_stdout=True)
   args = parser.parse_args()

   rx = radae_rx(args.model_name,auxdata=args.auxdata,v=args.v,disable_unsync=args.disable_unsync,foff_err=args.foff_err, bypass_dec=args.bypass_dec)

   # allocate storage for output features
   floats_out = np.zeros(rx.get_n_floats_out(),dtype=np.float32)
   while True:
      buffer = sys.stdin.buffer.read(rx.get_nin()*struct.calcsize("ff"))
      if len(buffer) != rx.get_nin()*struct.calcsize("ff"):
         break
      buffer_complex = np.frombuffer(buffer,np.csingle)
      valid_output = rx.do_radae_rx(buffer_complex, floats_out)
      if valid_output and args.use_stdout:
         sys.stdout.buffer.write(floats_out)
