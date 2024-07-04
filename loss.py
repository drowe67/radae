"""
/* Copyright (c) 2024 David Rowe */
   
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

from radae import distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to input feature file in .f32 format')
parser.add_argument('features_hat', type=str, help='path to output feature file in .f32 format')
parser.add_argument('--loss_test', type=float, default=0.0, help='compare loss to arg, print PASS/FAIL')
parser.add_argument('--acq_time_test', type=float, default=0, help='compare acquisition time to threshold arg, print PASS/FAIL')
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

features = load_features(args.features)
features_hat = load_features(args.features_hat)
features_seq_length = features.shape[1]
features_hat_seq_length = features_hat.shape[1]
print(features.shape, features_hat.shape)
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
print(f"loss: {min_loss:5.3f} start: {min_start:d} acq_time: {min_start*0.01:5.2f} s")
if args.loss_test > 0.0:
   if min_loss > args.loss_test:
      print("FAIL")
      quit()
if args.acq_time_test > 0:
   # one feature vector eevry 10ms
   if min_start*0.01 > args.acq_time_test:
      print("FAIL")
      quit()
if args.loss_test > 0.0 or args.acq_time_test:
   print("PASS")
