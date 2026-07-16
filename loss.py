"""
   Tool to measure loss between two feature files.

   Copyright (c) 2024 David Rowe
   
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
from matplotlib import pyplot as plt

from radae import distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--features_hat2', type=str, help='path to optional 2nd features file to compare two runs')
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to arg, print PASS/FAIL')
parser.add_argument('--acq_time_test', type=float, default=0, help='compare acquisition time to threshold arg, print PASS/FAIL')
parser.add_argument('--clip_start', type=int, default=0, help='remove this many feat vecs (e.g. frames x 4) from start (default 0)')
parser.add_argument('--clip_end', type=int, default=0, help='remove this many feat vecs (e.g. frames x 4) (default 0)')
parser.add_argument('--plot', action='store_true', help='plot loss versus time')
parser.add_argument('--png', type=str, default='', help='save plot to PNG file instead of displaying')
parser.add_argument('--stats', action='store_true', help='print per-frame loss statistics (mean, median, percentiles, outliers)')
parser.add_argument('--hist', action='store_true', help='plot histogram of per-frame loss distribution')
parser.add_argument('--outlier_threshold', type=float, default=0.0, help='loss threshold for outlier detection in --stats (default: 3x median)')
parser.add_argument('--compare', action='store_true', help='compare features_hat and features_hat2')
parser.add_argument('--delta', type=float, default=0.01, help='threshold for --compare')
args = parser.parse_args()

device = torch.device("cpu")
nb_total_features = 36
num_features = 20
num_used_features = 20
Tstep=0.01

def load_features(filename):
   features = np.reshape(np.fromfile(filename, dtype=np.float32), (1, -1, nb_total_features))
   features = features[:, :, :num_used_features]
   features = torch.tensor(features)
   return features

def find_loss(features_fn, features_hat_fn):
   features = load_features(features_fn)
   # zero pad either side to support +/- 1 second time alignment range
   pad_time = 1.
   pad = torch.zeros((1,int(pad_time/Tstep),num_used_features))
   features = torch.cat([pad,features,pad],dim=1)

   features_hat = load_features(features_hat_fn)
   features_hat = features_hat[:,args.clip_start:features_hat.shape[1]-args.clip_end,:]
   features_seq_length = features.shape[1]
   features_hat_seq_length = features_hat.shape[1]
   assert features_hat_seq_length
   if features_hat_seq_length >= features_seq_length:
      print(f"features_hat_length: {features_hat_seq_length:d} > features_length: {features_seq_length:d}")
      quit()
      
   # So features_hat will be shorter than features sequence.  Time align them based on min loss
   min_loss = distortion_loss(features[:,:features_hat_seq_length,:],features_hat).cpu().detach().numpy()[0]
   min_start = 0
   for start in range(features_seq_length-features_hat_seq_length):
      loss = distortion_loss(features[:,start:start+features_hat_seq_length,:],features_hat).cpu().detach().numpy()[0]
      if loss < min_loss:
         min_loss = loss
         min_start = start
   print(f"Loss between {features_fn:s} and {features_hat_fn:s}")
   acq_time = min_start*Tstep - pad_time
   print(f"  loss: {min_loss:5.3f} start: {min_start:d} acq_time: {acq_time:5.2f} s")

   # compute frame by frame loss for plotting
   nframes = features_hat_seq_length - min_start
   #print(min_start,nframes)
   loss = np.zeros(nframes)
   for f in range(nframes):
      loss[f] = distortion_loss(features[:,f+min_start:f+min_start+1,:],features_hat[:,f:f+1,:]).cpu().detach().numpy()[0]
   return min_loss, acq_time, loss

min_loss, acq_time, loss = find_loss(args.features, args.features_hat)

if args.loss_test > 0.0:
   if min_loss > args.loss_test:
      print("FAIL")
      quit()
if args.acq_time_test > 0:
   # one feature vector every 10ms
   if acq_time > args.acq_time_test:
      print("FAIL")
      quit()
if args.loss_test > 0.0 or args.acq_time_test:
   print("PASS")

if args.features_hat2:
   min_loss2, acq_time2, loss2 = find_loss(args.features, args.features_hat2)
   if args.compare:
      print(f"loss1: {min_loss:5.3f} loss2: {min_loss2:5.3f} delta: {np.abs(min_loss-min_loss2):5.3f}")
      if np.abs(min_loss-min_loss2) < args.delta:
         print("PASS")

if args.stats:
   def print_stats(loss_arr, label):
      threshold = args.outlier_threshold if args.outlier_threshold > 0.0 else 3.0 * np.median(loss_arr)
      outliers = np.sum(loss_arr > threshold)
      print(f"Stats for {label}:")
      print(f"  mean:   {np.mean(loss_arr):6.3f}  median: {np.median(loss_arr):6.3f}")
      print(f"  p95:    {np.percentile(loss_arr,95):6.3f}  p99:    {np.percentile(loss_arr,99):6.3f}")
      print(f"  max:    {np.max(loss_arr):6.3f}  outlier threshold: {threshold:6.3f}")
      print(f"  outliers (>{threshold:.3f}): {outliers:d} / {len(loss_arr):d} frames ({100*outliers/len(loss_arr):.1f}%)")
   print_stats(loss, args.features_hat)
   if args.features_hat2:
      print_stats(loss2, args.features_hat2)

if args.plot or args.png:
   plt.figure(1)
   t = np.arange(len(loss)) * Tstep
   plt.plot(t, loss, "b-", label=args.features_hat)
   if args.features_hat2:
      acq_timestep = int(acq_time2/Tstep)
      t2 = np.arange(len(loss2)) * Tstep + acq_time2
      plt.plot(t2, loss2, "r-", label=args.features_hat2)
   plt.xlabel('Time (s)'); plt.ylabel('Loss'); plt.grid()
   plt.legend(loc="upper right")
   if args.png:
      plt.savefig(args.png)
      print(f"Saved plot to {args.png}")
   else:
      plt.show()

if args.hist:
   plt.figure(2)
   threshold = args.outlier_threshold if args.outlier_threshold > 0.0 else 3.0 * np.median(loss)
   plt.hist(loss, bins=50, color='b', alpha=0.7, label=args.features_hat)
   if args.features_hat2:
      plt.hist(loss2, bins=50, color='r', alpha=0.7, label=args.features_hat2)
   plt.axvline(np.median(loss), color='b', linestyle='--', label=f'median {np.median(loss):.3f}')
   plt.axvline(threshold, color='k', linestyle=':', label=f'outlier threshold {threshold:.3f}')
   plt.xlabel('Loss'); plt.ylabel('Frame count'); plt.grid()
   plt.legend(loc="upper right")
   hist_png = args.png.replace('.png', '_hist.png') if args.png else ''
   if hist_png:
      plt.savefig(hist_png)
      print(f"Saved histogram to {hist_png}")
   else:
      plt.show()
