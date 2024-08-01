"""
/* Estimate CPU and memory requirements */

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

def gru(n_infeat, n_outfeat, multiplies, adds):
    # for one encoder timestep
    # Ref: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

    # step 1: r_t
    multiplies += 2*n_infeat*n_outfeat
    adds += 2*n_infeat*n_outfeat + 3*n_outfeat
  
    # step 2: z_t
    multiplies += 2*n_infeat*n_outfeat
    adds += 2*n_infeat*n_outfeat + 3*n_outfeat

    # step 3: n_t
    multiplies += 2*n_infeat*n_outfeat + n_outfeat
    adds += 2*n_infeat*n_outfeat + 3*n_outfeat

    # step 4: h_t
    multiplies += 2*n_outfeat
    adds += 2*n_outfeat
   
    # TODO sigmoid and tanh (treat as look ups?)

    return multiplies,adds

def conv1d(n_inchan, n_outchan, kernel_size, multiplies, adds):
    # for one encoder timestep
    multiplies += kernel_size*n_inchan*n_outchan
    adds += kernel_size*n_inchan*n_outchan + n_outchan

    return multiplies,adds

def linear(n_infeat, n_outfeat, multiplies, adds):
    multiplies += n_infeat*n_outfeat
    adds += n_infeat*n_outfeat + n_outfeat

    return multiplies,adds

if __name__ == '__main__':
    Tz = 0.04 # one encoder timestep

    gru_multiplies,gru_adds = gru(64,64,0,0)
    gru_multiplies,gru_adds = gru(224,64, gru_multiplies,gru_adds)
    gru_multiplies,gru_adds = gru(384,64, gru_multiplies,gru_adds)
    gru_multiplies,gru_adds = gru(544,64, gru_multiplies,gru_adds)
    gru_multiplies,gru_adds = gru(704,64, gru_multiplies,gru_adds)
    print(f"GRU per encoder call multiplies: {gru_multiplies:d} adds: {gru_adds}")

    conv1d_multiplies,conv1d_adds = conv1d(128,96, 2, 0, 0)
    conv1d_multiplies,conv1d_adds = conv1d(288,96, 2, conv1d_multiplies,conv1d_adds)
    conv1d_multiplies,conv1d_adds = conv1d(448,96, 2, conv1d_multiplies,conv1d_adds)
    conv1d_multiplies,conv1d_adds = conv1d(544,96, 2, conv1d_multiplies,conv1d_adds)
    conv1d_multiplies,conv1d_adds = conv1d(768,96, 2, conv1d_multiplies,conv1d_adds)
    print(f"Conv1D per encoder call multiplies: {conv1d_multiplies:d} adds: {conv1d_adds}")

    linear_multiplies,linear_adds = linear(4*22,64,0,0)
    linear_multiplies,linear_adds = linear(864,80,linear_multiplies,linear_adds)
    print(f"Linear per encoder call multiplies: {linear_multiplies:d} adds: {linear_adds}")

    mmacs = (gru_multiplies+conv1d_multiplies+linear_multiplies)/Tz/1E6
    print(f"CoreEncoder MMACs per second: {mmacs}")

