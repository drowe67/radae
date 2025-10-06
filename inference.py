"""
/* Copyright (c) 2024 modifications for radio autoencoder project
   by David Rowe */
   
/* Copyright (c) 2022 Amazon
   Written by Jan Buethe */
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
import torch

from radae import RADAE, distortion_loss,complex_bpf

parser = argparse.ArgumentParser()

parser.add_argument('model_name', type=str, help='path to model in .pth format')
parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--latent-dim', type=int, help="number of symbols produces by encoder, default: 80", default=80)
parser.add_argument('--frames_per_step', type=int, help="number of feat vecs per encoder/decoder vec, default: 4", default=4)
parser.add_argument('--cuda-visible-devices', type=str, help="set to 0 to run using GPU rather than CPU", default="")
parser.add_argument('--write_latent', type=str, default="", help='path to output file of latent vectors z[latent_dim] in .f32 format')
parser.add_argument('--EbNodB', type=float, default=100, help='BPSK Eb/No in dB')
parser.add_argument('--passthru', action='store_true', help='copy features in to feature out, bypassing ML network')
parser.add_argument('--mp_test', action='store_true', help='Fixed notch test multipath channel (rate Rs)')
parser.add_argument('--ber_test', action='store_true', help='send random PSK bits through channel model, measure BER')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath samples, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('--h_complex', action='store_true', help='use complex64 format samples in h_file (default mag only float32)')
parser.add_argument('--g_file', type=str, default="", help='path to rate Fs Doppler spread samples, ...G1G2G1G2... .f32 format')
parser.add_argument('--rate_Fs', action='store_true', help='rate Fs simulation (default rate Rs)')
parser.add_argument('--write_rx', type=str, default="", help='path to output file of rate Fs rx samples in ..IQIQ...f32 format')
parser.add_argument('--rx_gain', type=float, default=1.0, help='gain to apply to --write_rx samples (default 1.0)')
parser.add_argument('--write_tx', type=str, default="", help='path to output file of rate Fs tx samples in ..IQIQ...f32 format')
parser.add_argument('--phase_offset', type=float, default=0, help='phase offset in rads')
parser.add_argument('--freq_offset', type=float, default=0, help='freq offset in Hz')
parser.add_argument('--time_offset', type=int, default=0, help='sampling time offset in samples')
parser.add_argument('--df_dt', type=float, default=0, help='rate of change of freq offset in Hz/s')
parser.add_argument('--gain', type=float, default=1.0, help='rx gain (defaul 1.0)')
parser.add_argument('--pilots', action='store_true', help='insert pilot symbols')
parser.add_argument('--pilot_eq', action='store_true', help='use pilots to EQ data symbols using classical DSP')
parser.add_argument('--eq_ls', action='store_true', help='Use per carrier least squares EQ (default mean6)')
parser.add_argument('--cp', type=float, default=0.0, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0)')
parser.add_argument('--coarse_mag', action='store_true', help='Coarse magnitude correction (fixes --gain)')
parser.add_argument('--bottleneck', type=int, default=0, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to arg, print PASS/FAIL')
parser.add_argument('--prepend_noise', type=float, default=0.0, help='insert time (sec) of just rate Fs channel noise (no RADAE signal) at start (default 0)')
parser.add_argument('--append_noise', type=float, default=0.0, help='insert time (sec) of just rate Fs channel noise (no RADAE signal) at end (default 0)')
parser.add_argument('--end_of_over', action='store_true', help='insert end of over pilot sequence on last two modem frames (default off)')
parser.add_argument('--correct_freq_offset', action='store_true', help='correct --freq_offset before decoding here (default off)')
parser.add_argument('--sine_amp', type=float, default=0.0, help='single freq interferer level (default zero)')
parser.add_argument('--sine_freq', type=float, default=1000.0, help='single freq interferer freq (default 1000Hz)')
parser.add_argument('--auxdata', action='store_true', help='inject auxillary data symbol')
parser.add_argument('--txbpf', action='store_true', help='clipper/BPF styyle compressor')
parser.add_argument('--ssb_bpf', action='store_true', help=' SSB BPF simulation')
parser.add_argument('--pilots2', action='store_true', help='insert pilot symbols inside z vectors, replacing data symbols')
parser.add_argument('--correct_time_offset', type=int, default=0, help='introduces a delay (or advance if -ve) in samples, applied in freq domain (default 0)')
parser.add_argument('--tanh_clipper', action='store_true', help='use tanh magnitude clipper (default hard clipper)')
parser.add_argument('--w1_enc', type=int, default=64, help='Encoder GRU output dimension (default 64)')
parser.add_argument('--w2_enc', type=int, default=96, help='Encoder conv output dimension (default 96)')
parser.add_argument('--w1_dec', type=int, default=96, help='Decoder GRU output dimension (default 96)')
parser.add_argument('--w2_dec', type=int, default=32, help='Decoder conv output dimension (default 32)')
parser.add_argument('--peak', action='store_true', help='include peak power in loss function (alternative to bottleneck)')
args = parser.parse_args()

if len(args.h_file):
   if args.rate_Fs:
      print("WARNING: --g_file should be used to define the multipath model with --rate_Fs")

# set visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

latent_dim = args.latent_dim

# not exposed
nb_total_features = 36
num_features = 20
num_used_features = 20
if args.auxdata:
    num_features += 1

# load model from a checkpoint file
model = RADAE(num_features, latent_dim, args.EbNodB, ber_test=args.ber_test, rate_Fs=args.rate_Fs, 
              phase_offset=args.phase_offset, freq_offset=args.freq_offset, df_dt=args.df_dt,
              gain=args.gain, pilots=args.pilots, pilot_eq=args.pilot_eq, eq_mean6 = not args.eq_ls,
              cyclic_prefix = args.cp, time_offset=args.time_offset, coarse_mag=args.coarse_mag, 
              bottleneck=args.bottleneck, correct_freq_offset=args.correct_freq_offset, txbpf_en = args.txbpf,
              pilots2=args.pilots2, correct_time_offset=args.correct_time_offset, tanh_clipper=args.tanh_clipper,
              frames_per_step=args.frames_per_step, ssb_bpf = args.ssb_bpf, w1_dec=args.w1_dec, w2_dec=args.w2_dec,
              w1_enc=args.w1_enc, w2_enc=args.w2_enc, peak=args.peak)
checkpoint = torch.load(args.model_name, map_location='cpu',weights_only=True)
model.load_state_dict(checkpoint['state_dict'], strict=False)
checkpoint['state_dict'] = model.state_dict()

# dataloader
feature_file = args.features
features_in = np.reshape(np.fromfile(feature_file, dtype=np.float32), (1, -1, nb_total_features))
nb_features_rounded = model.num_10ms_times_steps_rounded_to_modem_frames(features_in.shape[1])
features = features_in[:,:nb_features_rounded,:]
features = features[:, :, :num_used_features]
if args.auxdata:
   aux_symb =  -np.ones((1,features.shape[1],1),dtype=np.float32)
   features = np.concatenate([features, aux_symb],axis=2)
features = torch.tensor(features)
print(f"Processing: {nb_features_rounded} feature vectors")

# default rate Rs multipath model H=1
Rs = model.Rs
Nc = model.Nc
num_timesteps_at_rate_Rs = model.num_timesteps_at_rate_Rs(nb_features_rounded)
H = torch.ones((1,num_timesteps_at_rate_Rs,Nc))

# construct a contrived rate Rs multipath model, will be a series of peaks an notches, between H=2 an H=0
if args.mp_test:
   G1 = 1
   G2 = 1
   d  = 0.002

   for c in range(Nc):
      omega = 2*np.pi*c
      arg = torch.tensor(-1j*omega*d*Rs)
      H[0,:,c] = torch.abs(G1 + G2*torch.exp(arg))  # in this case channel doesn't evolve over time
                                                    # only mag matters, we assume external phase equalisation

# user supplied rate Rs multipath model, sequence of H matrices
if args.h_file:
   if args.h_complex:
      h_dtype = np.complex64
   else:
      h_dtype = np.float32
   H = np.reshape(np.fromfile(args.h_file, dtype=h_dtype), (1, -1, Nc))
   if H.shape[1] < num_timesteps_at_rate_Rs:
      print("Multipath H file too short")
      quit()
   H = H[:,:num_timesteps_at_rate_Rs,:]
   H = torch.tensor(H)

# default rate Fs multipath model G1=1, G2=0
num_timesteps_at_rate_Fs = model.num_timesteps_at_rate_Fs(num_timesteps_at_rate_Rs)
G = torch.ones((1,num_timesteps_at_rate_Fs,2), dtype=torch.complex64)
G[:,:,1] = 0
# user supplied rate Fs multipath model, sequence of G1,G2 complex Doppler spread samples
if args.g_file:
   G = np.reshape(np.fromfile(args.g_file, dtype=np.csingle), (1, -1, 2))
   # first sample in file is estimate of gain required for a mean power of 1 through
   # the multipath channel over long runs, but in practice this is hard to predict as
   # test sample runs are short  
   mp_gain = np.real(G[:,0,0])
   G = mp_gain*G[:,1:,:] 
   if G.shape[1] < num_timesteps_at_rate_Fs:
      print("Multipath Doppler spread file too short")
      quit()
   G = G[:,:num_timesteps_at_rate_Fs,:]
   G = torch.tensor(G)


if __name__ == '__main__':

   if args.passthru:
      features_hat = features_in.flatten()
      features_hat.tofile(args.features_hat)
      quit()

   # push model to device and run test
   model.to(device)
   features = features.to(device)
   H = H.to(device)
   G = G.to(device)
   output = model(features,H,G)

   # target SNR calcs for a fixed Eb/No run (e.g. inference)
   EbNo = 10**(args.EbNodB/10)                # linear Eb/No
   B = 3000                                   # (kinda arbitrary) bandwidth for measuring noise power (Hz)
   SNR = EbNo*(model.Rb/B)
   SNRdB = 10*np.log10(SNR)
   CNodB = 10*np.log10(EbNo*model.Rb)
   print(f"          Eb/No   C/No     SNR3k  Rb'    Eq     PAPR")
   print(f"Target..: {args.EbNodB:6.2f}  {CNodB:6.2f}  {SNRdB:6.2f}  {int(model.Rb_dash):d}")

   # Lets check actual Eq/No, Eb/No and SNR, and monitor assumption |z| ~ 1, especially for multipath.
   # If |z| ~ 1, Eb ~ 1, Eq ~ 2, and the measured SNR should match the set point SNR. 
   if args.rate_Fs:
      # rate Fs simulation
      tx = output["tx"].cpu().detach().numpy()
      S = np.mean(np.abs(tx)**2)
      N = output["sigma"]**2                                 # noise power in B=Fs
      N = N.item()
      #print(S, N)
      CNodB_meas = 10*np.log10(S*model.Fs/N)                 # S/N = S/(NoB) = S/(NoFs), C = S, C/No = SFs/N
      # With a CP the Eb gets tricky, not simply C/No/Rb_dash.  Only M samples out of M+Ncp are used for detection, 
      # the Ncp samples used for the CP is discarded.  So we work Es=M/Fs for one OFDM symbol,
      # then divide by (Nc and bps) to get Eb. 
      EbNodB_meas = CNodB_meas + 10*np.log10(model.M/(model.Fs*model.Nc*model.bps))
      SNRdB_meas = CNodB_meas - 10*np.log10(B)               # SNR in B=3000
      PAPRdB = 20*np.log10(np.max(np.abs(tx))/np.sqrt(S))
      print(f"Measured: {EbNodB_meas:6.2f}  {CNodB_meas:6.2f}  {SNRdB_meas:6.2f}                {PAPRdB:5.2f}")
   else:
      # rate Rs simulation
      tx_sym = output["tx_sym"].cpu().detach().numpy()
      Eq_meas = np.mean(np.abs(tx_sym)**2)
      No = output["sigma"]**2
      No = No.item()
      EqNodB_meas = 10*np.log10(Eq_meas/No)
      Rq = Rs*Nc
      SNRdB_meas = EqNodB_meas + 10*np.log10(Rq/B)
      #print(EqNodB_meas,SNRdB_meas,Eq_meas)
      if model.bottleneck == 3:
         tx = output["tx"].cpu().detach().numpy()
         S = np.mean(np.abs(tx)**2)
         PAPRdB = 20*np.log10(np.max(np.abs(tx))/np.sqrt(S))
         print(f"Measured: {EqNodB_meas-3:6.2f}          {SNRdB_meas:6.2f}       {Eq_meas:7.2f} {PAPRdB:5.2f}")
      else:
         print(f"Measured: {EqNodB_meas-3:6.2f}          {SNRdB_meas:6.2f}       {Eq_meas:7.2f}")

   features_hat = output["features_hat"][:,:,:num_used_features]
   features_hat = torch.cat([features_hat, torch.zeros_like(features_hat)[:,:,:16]], dim=-1)
   features_hat = features_hat.cpu().detach().numpy().flatten().astype('float32')
   features_hat.tofile(args.features_hat)

   loss = distortion_loss(features[..., :20],output['features_hat'][..., :20]).cpu().detach().numpy()[0]
   if args.auxdata:
      x = features[..., 20:21]*output["features_hat"][..., 20:21]
      x = torch.flatten(x)
      n_errors = int(torch.sum(x < 0))
      n_bits = int(torch.numel(x))
      BER = n_errors/n_bits
      print(f"loss: {loss:5.3f} Auxdata BER: {BER:5.3f}")
   else:
      print(f"loss: {loss:5.3f}")
   if args.loss_test > 0.0:
      if loss < args.loss_test:
         print("PASS")
      else:
         print("FAIL")

   # write real valued latent vectors
   if len(args.write_latent):
      z_hat = output["z_hat"].cpu().detach().numpy().flatten().astype('float32')
      z_hat.tofile(args.write_latent)
   
   # write complex valued rate Fs time domain rx samples
   if len(args.write_rx):
      if args.rate_Fs:
         rx = output["rx"]
         sigma = output["sigma"].item()
         
         if args.end_of_over:
            # appends a frame containing a final pilot so the last RADAE frame
            # has a good phase reference, and two "end of over" symbols
            eoo = model.eoo

            # this is messy! - continue phase, freq and dF/dt track from inside forward()
            freq = torch.zeros_like(eoo)
            freq[:,] = model.freq_offset*torch.ones_like(eoo) + model.df_dt*torch.arange(eoo.shape[1])/model.Fs
            omega = freq*2*torch.pi/model.Fs
            lin_phase = torch.cumsum(omega,dim=1)
            lin_phase = torch.exp(1j*lin_phase)
            eoo = eoo*lin_phase*model.final_phase
            eoo = eoo + sigma*torch.randn_like(eoo)
            rx = torch.concatenate([rx,eoo],dim=1)
         if args.prepend_noise > 0.0:
            num_noise = int(model.Fs*args.prepend_noise)
            n = sigma*torch.randn(1,num_noise)
            rx = torch.concatenate([n,rx],dim=1)
         if args.append_noise > 0.0:
            num_noise = int(model.Fs*args.append_noise)
            n = sigma*torch.randn(1,num_noise)
            rx = torch.concatenate([rx,n],dim=1)
         if args.sine_amp > 0.0:
            s = args.sine_amp*torch.exp(1j*torch.arange(rx.shape[1])*2*torch.pi*args.sine_freq/model.Fs)
            print(rx.shape, s.shape,)
            rx[0,:] += s
         rx = args.rx_gain*rx.cpu().detach().numpy().flatten().astype('csingle')
         rx.tofile(args.write_rx)
      else:
         print("\nWARNING: Need --rate_Fs for --write_rx")
      
   # write complex valued rate Fs time domain tx samples
   if len(args.write_tx):
      if args.bottleneck == 3:
         tx = output["tx"].cpu().detach().numpy().flatten().astype('csingle')
         tx.tofile(args.write_tx)
      else:
         print("\nWARNING: Need --bottleneck 3 for --write_tx")
   

