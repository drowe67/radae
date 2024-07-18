"""

  Radio Autoencoder streaming receiver: 
  
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

import os, sys, argparse, struct
import numpy as np
from matplotlib import pyplot as plt
import torch
from radae import RADAE,complex_bpf,acquisition,receiver_one

parser = argparse.ArgumentParser(description='RADAE streaming receiver, IQ.f32 on stdin to features.f32 on stdout')

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('--rxfile', type=argparse.FileType("rb"), default=sys.stdin, help='path to input file of rate Fs rx samples in ..IQIQ...f32 format (default stdin)')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--ber_test', type=str, default="", help='symbols are PSK bits, compare to z.f32 file to calculate BER')
parser.add_argument('--no_bpf', action='store_false', dest='bpf', help='disable BPF')
parser.add_argument('--bottleneck', type=int, default=3, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--write_Dt', type=str, default="", help='Write D(t,f) matrix on last modem frame')
parser.add_argument('--acq_test',  action='store_true', help='Acquisition test mode')
parser.add_argument('--fmax_target', type=float, default=0.0, help='Acquisition test mode freq offset target (default 0.0)')
parser.add_argument('-v', type=int, default=2, help='Verbose level (default 2)')
parser.set_defaults(bpf=True)
args = parser.parse_args()

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

latent_dim = args.latent_dim
nb_total_features = 36
num_features = 20
num_used_features = 20

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, EbNodB=100, ber_test=args.ber_test, rate_Fs=True, 
              pilots=True, pilot_eq=True, eq_mean6 = False, cyclic_prefix=0.004,
              coarse_mag=True,time_offset=-16, bottleneck=args.bottleneck)
checkpoint = torch.load(args.model_name, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)
# Stateful decoder wasn't present during training, so we need to load weights from existing decoder
model.core_decoder_statefull_load_state_dict()

model.to(device) # TODO do we need this?

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

if args.bpf:
   Ntap=101
   bandwidth = 1.2*(w[Nc-1] - w[0])*model.Fs/(2*np.pi)
   centre = (w[Nc-1] + w[0])*model.Fs/(2*np.pi)/2
   print(f"Input BPF bandwidth: {bandwidth:f} centre: {centre:f}", file=sys.stderr)
   bpf = complex_bpf(Ntap, model.Fs, bandwidth,centre)

acq = acquisition(Fs,Rs,M,Ncp,Nmf,p,model.pend)

"""
# optional acq_test variables 
tmax_candidate_target = Ncp + Ntap/2
acq_pass = 0
acq_fail = 0
"""

tmax_candidate = 0 
acquired = False
state = "search"
prev_state = state
mf = 1
valid_count = 0
Tunsync = 3.0                        # allow some time before lossing sync to ride over fades
Nmf_unsync = int(Tunsync*Fs/Nmf)
endofover = False

rx_buf = np.zeros(2*Nmf+M+Ncp,np.csingle)
rx = np.zeros(0,np.csingle)
rx_phase = 1 + 1j*0
rx_phase_vec = np.zeros(Nmf+M+Ncp,np.csingle)
z_hat_log = torch.zeros(0,model.Nzmf,model.latent_dim)

while True:
   buffer = sys.stdin.buffer.read(Nmf*struct.calcsize("ff"))
   if len(buffer) != Nmf*struct.calcsize("ff"):
      break
   buffer_complex = np.frombuffer(buffer,np.csingle)
   if args.bpf:
      buffer_complex = bpf.bpf(buffer_complex)
   rx_buf[:Nmf+M+Ncp] = rx_buf[Nmf:]                           # out with the old
   rx_buf[Nmf+M+Ncp:] = buffer_complex                         # in with the new
   if state == "search" or state == "candidate":
      candidate, tmax, fmax = acq.detect_pilots(rx_buf)
   else:
      # we're in sync, so check we can still see pilots and run receiver
      ffine_range = np.arange(fmax-0.5,fmax+0.5,0.1)
      tfine_range = np.arange(tmax-1,tmax+2)
      tmax,fmax_hat = acq.refine(rx_buf, tmax, fmax, tfine_range, ffine_range)
      fmax = 0.9*fmax + 0.1*fmax_hat
      candidate,endofover = acq.check_pilots(rx_buf,tmax,fmax)
      if not endofover:
         # correct frequency offset, note we preserve state of phase
         # TODO do we need preserve state of phase?  We're passing entire vector and there isn't any memory (I think)
         w = 2*np.pi*fmax/Fs
         for n in range(Nmf+M+Ncp):
            rx_phase = rx_phase*np.exp(-1j*w)
            rx_phase_vec[n] = rx_phase
         rx = torch.tensor(rx_buf[tmax-Ncp:tmax-Ncp+Nmf+M+Ncp]*rx_phase_vec, dtype=torch.complex64)
         # run through RADAE receiver DSP
         z_hat = receiver.receiver_one(rx)
         # decode z_hat to features
         assert(z_hat.shape[1] == model.Nzmf)
         features_hat = torch.zeros(1,model.dec_stride*z_hat.shape[1],model.feature_dim)
         for i in range(model.Nzmf):
            features_hat[0,i*model.dec_stride:(i+1)*model.dec_stride,:] = model.core_decoder_statefull(z_hat[:,i:i+1,:])
         # add unused features and send to stdout
         features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
         features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
         sys.stdout.buffer.write(features_hat)
         #sys.stdout.flush()
         if len(args.write_latent):
            z_hat_log = torch.cat([z_hat_log,z_hat])

   if args.v == 2 or (args.v == 1 and (state == "search" or state == "candidate" or prev_state == "candidate")):
      print(f"{mf:3d} state: {state:10s} valid: {candidate:d} {endofover:d} {valid_count:2d} Dthresh: {acq.Dthresh:8.2f} ",end='', file=sys.stderr)
      print(f"Dtmax12: {acq.Dtmax12:8.2f} {acq.Dtmax12_eoo:8.2f} tmax: {tmax:4d} tmax_candidate: {tmax_candidate:4d} fmax: {fmax:6.2f}", file=sys.stderr)

   # iterate state machine  
   next_state = state
   prev_state = state
   if state == "search":
      if candidate:
         next_state = "candidate"
         tmax_candidate = tmax
         valid_count = 1
   elif state == "candidate":
      # look for 3 consecutive matches with about the same timing offset  
      if candidate and np.abs(tmax-tmax_candidate) < 0.02*M:
         valid_count = valid_count + 1
         if valid_count > 3:
            next_state = "sync"
            acquired = True
            valid_count = Nmf_unsync
            ffine_range = np.arange(fmax-10,fmax+10,0.25)
            tfine_range = np.arange(tmax-1,tmax+2)
            tmax,fmax = acq.refine(rx_buf, tmax, fmax, tfine_range, ffine_range)
      else:
         next_state = "search"
   elif state == "sync":
      if candidate:
         valid_count = Nmf_unsync
      else:
         valid_count -= 1
         if valid_count == 0:
            next_state = "search"
      if endofover:
         next_state = "search"
   state = next_state
   mf += 1


if not acquired:
   print("Acquisition failed....", file=sys.stderr)
   quit()

if len(args.write_latent) or len(args.ber_test):
   z_hat = z_hat_log.cpu().detach().numpy().flatten().astype('float32')

   # BER test useful for calibrating link.  To measure BER we compare the received symnbols 
   # to the known transmitted symbols.  However due to acquisition delays we may have lost several
   # modem frames in the received sequence.
   if len(args.ber_test):
      # every time acq shifted Nmf (one modem frame of samples), we shifted this many latents:
      num_latents_per_modem_frame = model.Nzmf*model.latent_dim
      #print(num_latents_per_modem_frame)
      z = np.fromfile(args.ber_test, dtype=np.float32)
      #print(z.shape, z_hat.shape)
      best_BER = 1
      # to find best alignment look for lowerest BER over a range of shifts
      for f in np.arange(20):
         n_syms = min(len(z),len(z_hat))
         n_errors = np.sum(-z[:n_syms]*z_hat[:n_syms]>0)
         n_bits = len(z)
         BER = n_errors/n_bits
         if BER < best_BER:
            best_BER = BER
            print(f"f: {f:2d} n_bits: {n_bits:d} n_errors: {n_errors:d} BER: {BER:5.3f}", file=sys.stderr)
         z = z[num_latents_per_modem_frame:]

   # write real valued latent vectors
   if len(args.write_latent):
      z_hat.tofile(args.write_latent)
