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
parser.add_argument('--compare', action='store_true', help='compare features_hat and features_hat2')
args = parser.parse_args()

device = torch.device("cpu")
nb_total_features = 36
num_features = 20
num_used_features = 20

def load_features(filename):
   features = np.reshape(np.fromfile(filename, dtype=np.float32), (1, -1, nb_total_features))
   features = features[:, :, :num_used_features]
   features = torch.tensor(features)
   return features

def find_loss(features_fn, features_hat_fn):
   features = load_features(features_fn)
   features_hat = load_features(features_hat_fn)
   features_hat = features_hat[:,args.clip_start:features_hat.shape[1]-args.clip_end,:]
   features_seq_length = features.shape[1]
   features_hat_seq_length = features_hat.shape[1]
   #print(features.shape, features_hat.shape)
   assert features_hat_seq_length
   assert features_hat_seq_length <= features_seq_length

   # So features_hat will be shorter than features sequence.  Time align them based on min loss
   min_loss = distortion_loss(features[:,:features_hat_seq_length,:],features_hat).cpu().detach().numpy()[0]
   min_start = 0
   for start in range(features_seq_length-features_hat_seq_length):
      loss = distortion_loss(features[:,start:start+features_hat_seq_length,:],features_hat).cpu().detach().numpy()[0]
      if loss < min_loss:
         min_loss = loss
         min_start = start
   print(f"Loss between {features_fn:s} and {features_hat_fn:s}")
   print(f"  loss: {min_loss:5.3f} start: {min_start:d} acq_time: {min_start*0.01:5.2f} s")

   # compute frame by frame loss for plotting
   nframes = features_hat_seq_length - min_start
   #print(min_start,nframes)
   loss = np.zeros(nframes)
   for f in range(nframes):
      loss[f] = distortion_loss(features[:,f+min_start:f+min_start+1,:],features_hat[:,f:f+1,:]).cpu().detach().numpy()[0]
   return min_loss, min_start, loss

min_loss, min_start,loss = find_loss(args.features, args.features_hat)

if args.loss_test > 0.0:
   if min_loss > args.loss_test:
      print("FAIL")
      quit()
if args.acq_time_test > 0:
   # one feature vector every 10ms
   if min_start*0.01 > args.acq_time_test:
      print("FAIL")
      quit()
if args.loss_test > 0.0 or args.acq_time_test:
   print("PASS")

if args.features_hat2:
   min_loss2, min_start2, loss2 = find_loss(args.features, args.features_hat2)
   if args.compare:
      print(f"loss1: {min_loss:5.3f} loss2: {min_loss2:5.3f} delta: {np.abs(min_loss-min_loss2):5.3f}")
      if np.abs(min_loss-min_loss2) < 0.01:
         print("PASS")

if args.plot:
   if args.features_hat2:
      plt.figure(1)
      plt.plot(loss, "b-", label=args.features_hat)
      x = range(min_start2,min_start2+len(loss2))
      plt.plot(x, loss2, "r-", label=args.features_hat2)
      plt.legend(loc="upper left")
      plt.figure(2)
      ax = (0,len(loss),0,max(max(loss),max(loss2)))
      plt.subplot(211)
      plt.plot(loss, "b-", label=args.features_hat)
      plt.axis(ax)
      plt.legend(loc="upper left")
      plt.subplot(212)
      plt.plot(x, loss2, "r-", label=args.features_hat2)
      plt.axis(ax)
      plt.legend(loc="upper left")
   else:
      plt.plot(loss, "b-", label=args.features_hat)
   plt.show()
