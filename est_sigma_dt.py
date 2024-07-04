"""
/*
  Prototyping code to test estimation of std dev of |Dt| from
  std dev of received signal. Ref "Pilot Tracking" section of
  radae.pdf
  
  Copyright (c) 2024 by David Rowe */

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
from matplotlib import pyplot as plt
import torch
from radae import RADAE

# make sure we don't use a GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
device = torch.device("cpu")

# Bring up a RADAE model just to generate the pilot sequence p 
latent_dim = 40
num_features = 20
model = RADAE(num_features, latent_dim, EbNodB=100, rate_Fs=True, pilots=True, cyclic_prefix=0.004)

M = model.M
p_cp = np.array(model.p_cp)
p = p_cp[model.Ncp:]
sigma_p = np.sqrt(np.dot(np.conj(p),p).real)

# Generate some Dt samples so we can measure std dev of |Dt|
# We assume received signal is just AWGN noise
Nsam = 1000
N = 100
sigma_rx = np.sqrt(N)
Dt=np.zeros(Nsam,dtype=np.csingle)
for i in range(Nsam):
   n = sigma_rx*(np.random.normal(size=M) + 1j*np.random.normal(size=M))/(2**0.5)
   r = n
   Dt[i] = np.dot(np.conj(r),p)

# now try approximation of |Dt|

sigma_dt = np.std(np.abs(Dt))
sigma_dt_est = sigma_rx*sigma_p/np.sqrt(5)
error = sigma_dt/sigma_dt_est
print(f"Std(|Dt|): {sigma_dt:f} {sigma_dt_est:f} Error: {error:f} sigma_rx: {sigma_rx:f} sigma_p: {sigma_p:f}")

sigma_r = np.mean(np.abs(Dt))/((np.pi/2)**0.5)
sigma_r_est = sigma_rx*sigma_p/np.sqrt(10.0-5*np.pi/2)
print(f"sigma_r: {sigma_r:f} {sigma_r_est:f}")
