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
import torch
import tqdm
from matplotlib import pyplot as plt
import numpy as np
import sys

from radae import RADAE, RADAEDataset, distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output folder')
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: ''", default="")
parser.add_argument('--latent-dim', type=int, help="number of symbols produced by encoder, default: 80", default=80)
parser.add_argument('--EbNodB', type=float, default=0, help='BPSK Eb/No in dB')
parser.add_argument('--range_EbNo', action='store_true', help='Use a range of Eb/No during training')
parser.add_argument('--range_EbNo_start', type=float, default=-6.0, help='starting value for Eb/No during training')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath file, rate Rs time steps by Nc carriers .f32 format')
parser.add_argument('--g_file', type=str, default="", help='path to rate Fs multipath file, ...G1G2... .f32 format')
parser.add_argument('--rate_Fs', action='store_true', help='rate Fs simulation (default rate Rs)')
parser.add_argument('--freq_rand', action='store_true', help='random phase and freq offset for each sequence')
parser.add_argument('--gain_rand', action='store_true', help='random rx gain -20 .. +20dB, SNR unchanged')
parser.add_argument('--bottleneck', type=int, default=1, help='1-1D rate Rs, 2-2D rate Rs, 3-2D rate Fs time domain')
parser.add_argument('--pilots', action='store_true', help='insert pilot symbols')
parser.add_argument('--pilot_eq', action='store_true', help='use pilots to EQ data symbols using classical DSP')
parser.add_argument('--eq_ls', action='store_true', help='Use per carrier least squares EQ (default mean6)')
parser.add_argument('--cp', type=float, default=0.0, help='Length of cyclic prefix in seconds [--Ncp..0], (default 0)')

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 32", default=32)
training_group.add_argument('--lr', type=float, help='learning rate, default: 3e-4', default=3e-4)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 100', default=100)
training_group.add_argument('--sequence-length', type=int, help='sequence length, needs to be divisible by 4, default: 256', default=256)
training_group.add_argument('--lr-decay-factor', type=float, help='learning rate decay factor, default: 2.5e-5', default=2.5e-5)

training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
training_group.add_argument('--plot_loss', action='store_true', help='plot loss versus epoch as we train')
training_group.add_argument('--plot_EqNo', type=str, default="", help='plot loss versus Eq/No for final epoch')
training_group.add_argument('--auxdata', action='store_true', help='inject auxillary data symbol')

args = parser.parse_args()

# set visible devices
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices

# checkpoints
checkpoint_dir = os.path.join(args.output, 'checkpoints')
checkpoint = dict()
os.makedirs(checkpoint_dir, exist_ok=True)

# training parameters
batch_size = args.batch_size
lr = args.lr
epochs = args.epochs
sequence_length = args.sequence_length
lr_decay_factor = args.lr_decay_factor

# not exposed
adam_betas = [0.8, 0.95]
adam_eps = 1e-8

checkpoint['batch_size'] = batch_size
checkpoint['lr'] = lr
checkpoint['lr_decay_factor'] = lr_decay_factor
checkpoint['epochs'] = epochs
checkpoint['sequence_length'] = sequence_length
checkpoint['adam_betas'] = adam_betas

# logging
log_interval = 10

# device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")

# model parameters
latent_dim = args.latent_dim

num_features = 20
if args.auxdata:
    num_features += 1

# training data
feature_file = args.features

# model
checkpoint['model_args'] = (num_features, latent_dim, args.EbNodB, args.range_EbNo, args.rate_Fs)
model = RADAE(num_features, latent_dim, args.EbNodB, range_EbNo=args.range_EbNo, 
              rate_Fs = args.rate_Fs,
              range_EbNo_start=args.range_EbNo_start, 
              freq_rand=args.freq_rand,gain_rand=args.gain_rand, bottleneck=args.bottleneck,
              pilots=args.pilots, pilot_eq=args.pilot_eq, eq_mean6 = not args.eq_ls, cyclic_prefix = args.cp)

if type(args.initial_checkpoint) != type(None):
    print(f"Loading from checkpoint: {args.initial_checkpoint}")
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint['state_dict']    = model.state_dict()

# dataloader
Nc = model.Nc
H_sequence_length = model.num_timesteps_at_rate_Rs(sequence_length)
G_sequence_length = model.num_timesteps_at_rate_Fs(H_sequence_length)

checkpoint['dataset_args'] = (feature_file, sequence_length, H_sequence_length, Nc, G_sequence_length)
checkpoint['dataset_kwargs'] = {'enc_stride': model.enc_stride}
dataset = RADAEDataset(*checkpoint['dataset_args'], h_file = args.h_file, g_file = args.g_file, auxdata=args.auxdata)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=lr, betas=adam_betas, eps=adam_eps)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay_factor * x))

if __name__ == '__main__':

    # push model to device
    model.to(device)
    model.move_device(device)

    # -----------------------------------------------------------------------------------------------
    # run through dataset once with current model but training disabled, to gather loss v EqNo stats
    # -----------------------------------------------------------------------------------------------
    if len(args.plot_EqNo):
        # TODO: move this to a function
        print("Measuring loss ve Eq/No over training set with training disabled")
        model.eval()
        EqNodB_loss = np.zeros((dataloader.__len__()*batch_size,2))
        running_total_loss      = 0
        previous_total_loss     = 0
        current_loss            = 0.

        with torch.no_grad():
            with tqdm.tqdm(dataloader, unit='batch') as tepoch:
                for i, (features,H,G) in enumerate(tepoch):
                    features = features.to(device)
                    H = H.to(device)
                    G = G.to(device)
                    output = model(features,H,G)
                    loss_by_batch = distortion_loss(features[..., :20], output["features_hat"][..., :20])
                    total_loss = torch.mean(loss_by_batch)
                    
                    # collect running Eq/No stats, measured Eq/No and loss for each sequence in batch
                    if args.rate_Fs:
                        tx = output["tx"].cpu().detach().numpy()
                        S = np.mean(np.abs(tx)**2,axis=1)
                        N = output["sigma"][:,0]**2                          # noise power in B=Fs
                        CNodB_meas = 10*np.log10(S*model.Fs/N)               # S/N = S/(NoB) = S/(NoFs), C = S, C/No = SFs/N
                        EqNodB_meas = CNodB_meas - 10*np.log10(model.Rs*model.Nc)
                    else:
                        tx_sym = output["tx_sym"].cpu().detach().numpy()
                        Eq_meas = np.mean(np.abs(tx_sym)**2,axis=(1,2))
                        No = output["sigma"][:,0,0]**2
                        EqNodB_meas = 10*np.log10(Eq_meas/No)
                    EqNodB_loss[i*batch_size:(i+1)*batch_size,0] = EqNodB_meas
                    EqNodB_loss[i*batch_size:(i+1)*batch_size,1] = loss_by_batch.cpu().detach().numpy()                       

                    running_total_loss += float(total_loss.detach().cpu())
                    
                    if (i + 1) % log_interval == 0:
                        current_loss = (running_total_loss - previous_total_loss) / log_interval
                        tepoch.set_postfix(
                            current_loss=current_loss,
                            total_loss=running_total_loss / (i + 1),
                        )
                        previous_total_loss = running_total_loss

        # Plot loss against EqNodB for final epoch, using log of loss and Eq/No for
        # each sequence.  We group losses into 1dB Eq/No bins, kind of like a histogram.
        EqNodB_min = int(np.ceil(np.min(EqNodB_loss[:,0])))
        EqNodB_max = int(np.ceil(np.max(EqNodB_loss[:,0])))
        EqNodB_mean_loss = np.zeros((EqNodB_max-EqNodB_min,2))
        # group the losses from training into 1dB wide bins, and find mean for that bin
        r = np.arange(EqNodB_min,EqNodB_max)
        for i in np.arange(len(r)):
            EqNodB = r[i]
            x = np.where(np.abs(EqNodB_loss[:,0] - EqNodB) < 0.5)
            EqNodB_mean_loss[i,0] = EqNodB
            EqNodB_mean_loss[i,1] = np.mean(EqNodB_loss[x,1])
        plt.figure(2)
        plt.plot(EqNodB_mean_loss[:,0],EqNodB_mean_loss[:,1],'b+-')
        plt.grid()
        plt.xlabel('Eq/No (dB)')
        plt.ylabel('Loss')
        plt.show(block=False)
        plt.savefig(args.plot_EqNo + '_loss_EqNodB.png')

        np.savetxt(args.plot_EqNo + '_loss_EqNodB' + '.txt', EqNodB_mean_loss)
        quit()

    # ---------------------------------------------------------------------------------------------
    # Regular training loop
    # ---------------------------------------------------------------------------------------------

    if args.plot_loss:
        plt.figure(1)
        loss_epoch=np.zeros((args.epochs+1))
    
    # Main training loop
    for epoch in range(1, epochs + 1):

        print(f"training epoch {epoch}...",file=sys.stderr)

        # running stats
        running_total_loss      = 0
        previous_total_loss     = 0
        current_loss            = 0.

        if args.plot_loss:
            plt.figure(1)
            
        with tqdm.tqdm(dataloader, unit='batch') as tepoch:
            for i, (features,H,G) in enumerate(tepoch):

                optimizer.zero_grad()
                features = features.to(device)
                H = H.to(device)
                if len(args.g_file):
                    G = G.to(device)
                    output = model(features,H,G)
                else:
                    output = model(features,H)
                loss_by_batch = distortion_loss(features, output["features_hat"])
                total_loss = torch.mean(loss_by_batch)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                running_total_loss += float(total_loss.detach().cpu())
                
                if (i + 1) % log_interval == 0:
                    current_loss = (running_total_loss - previous_total_loss) / log_interval
                    if args.auxdata:
                        # sample latest BER
                        x = features[..., 20:21]*output["features_hat"][..., 20:21]
                        x = torch.flatten(x)
                        n_errors = int(torch.sum(x < 0))
                        n_bits = int(torch.numel(x))
                        BER = n_errors/n_bits
                        tepoch.set_postfix(
                            current_loss=current_loss,
                            total_loss=running_total_loss / (i + 1),
                            BER=BER
                        )
                    else:
                        tepoch.set_postfix(
                            current_loss=current_loss,
                            total_loss=running_total_loss / (i + 1),
                        )
                    previous_total_loss = running_total_loss
                    if args.plot_loss:
                        loss_epoch[epoch] = current_loss

        if args.plot_loss:
            plt.clf()
            plt.semilogy(range(1,epoch+1),loss_epoch[1:epoch+1])
            plt.grid()
            plt.show(block=False)
            plt.pause(0.01)

        # save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        checkpoint['state_dict'] = model.state_dict()
        checkpoint['loss'] = running_total_loss / len(dataloader)
        checkpoint['epoch'] = epoch
        torch.save(checkpoint, checkpoint_path)

        if args.plot_loss:
            plt.savefig(args.output + '_loss.png')
            np.savetxt(args.output + '_loss' + '.txt', loss_epoch[1:epoch+1])

