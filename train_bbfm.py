"""

   Training Baseband FM version of Radio Autoencoder

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

from radae import BBFM, RADAEDataset, distortion_loss

parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='path to feature file in .f32 format')
parser.add_argument('output', type=str, help='path to output folder')
parser.add_argument('--cuda-visible-devices', type=str, help="comma separates list of cuda visible device indices, default: ''", default="")
parser.add_argument('--latent-dim', type=int, help="number of symbols produced by encoder, default: 80", default=80)
parser.add_argument('--RdBm', type=float, default=-100.0, help='Receive level set point in dBm (default -120)')
parser.add_argument('--range_RdBm',  action='store_true', help='Sweep receive level during training')
parser.add_argument('--h_file', type=str, default="", help='path to rate Rs multipath file, rate Rs time steps by 1 carriers .f32 format')

training_group = parser.add_argument_group(title="training parameters")
training_group.add_argument('--batch-size', type=int, help="batch size, default: 32", default=32)
training_group.add_argument('--lr', type=float, help='learning rate, default: 3e-4', default=3e-4)
training_group.add_argument('--epochs', type=int, help='number of training epochs, default: 100', default=100)
training_group.add_argument('--sequence-length', type=int, help='sequence length, needs to be divisible by 4, default: 256', default=256)
training_group.add_argument('--lr-decay-factor', type=float, help='learning rate decay factor, default: 2.5e-5', default=2.5e-5)

training_group.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
training_group.add_argument('--plot_loss', action='store_true', help='plot loss versus epoch as we train')
training_group.add_argument('--plot_R', type=str, default="", help='plot loss versus RdBm for final epoch, arg is suffix')

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

# training data
feature_file = args.features

# model
checkpoint['model_args'] = (num_features, latent_dim, args.RdBm)
model = BBFM(num_features, latent_dim, args.RdBm, range_RdBm=args.range_RdBm)

if type(args.initial_checkpoint) != type(None):
    print(f"Loading from checkpoint: {args.initial_checkpoint}")
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

checkpoint['state_dict']    = model.state_dict()

# dataloader
Nc = 1
H_sequence_length = model.num_timesteps_at_rate_Rs(sequence_length)
G_sequence_length = 0

checkpoint['dataset_args'] = (feature_file, sequence_length, H_sequence_length, Nc, G_sequence_length)
checkpoint['dataset_kwargs'] = {'enc_stride': model.enc_stride}
dataset = RADAEDataset(*checkpoint['dataset_args'], h_file = args.h_file)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=lr, betas=adam_betas, eps=adam_eps)

# learning rate scheduler
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x : 1 / (1 + lr_decay_factor * x))

if __name__ == '__main__':

    # push model to device
    model.to(device)

    # -----------------------------------------------------------------------------------------------
    # run through dataset once with current model but training disabled, to gather loss v R stats
    # -----------------------------------------------------------------------------------------------
    if len(args.plot_R):
        # TODO: move this to a function
        print("Measuring loss v RdBm over training set with training disabled")
        model.eval()
        R_loss = np.zeros((dataloader.__len__()*batch_size,2))
        running_total_loss      = 0
        previous_total_loss     = 0
        current_loss            = 0.

        with torch.no_grad():
            with tqdm.tqdm(dataloader, unit='batch') as tepoch:
                for i, (features,H,G) in enumerate(tepoch):
                    features = features.to(device)
                    H = H.to(device)
                    output = model(features,H)
                    loss_by_batch = distortion_loss(features[..., :20], output["features_hat"][..., :20])
                    total_loss = torch.mean(loss_by_batch)
                    
                    # collect running stats, R and loss for each sequence in batch
                    R_loss[i*batch_size:(i+1)*batch_size,0] = output["RdBm_"].cpu().detach().numpy()
                    R_loss[i*batch_size:(i+1)*batch_size,1] = loss_by_batch.cpu().detach().numpy()                       

                    running_total_loss += float(total_loss.detach().cpu())
                    
                    if (i + 1) % log_interval == 0:
                        current_loss = (running_total_loss - previous_total_loss) / log_interval
                        tepoch.set_postfix(
                            current_loss=current_loss,
                            total_loss=running_total_loss / (i + 1),
                        )
                        previous_total_loss = running_total_loss

        # Plot loss against EqNodB for final epoch, using log of loss and Eq/No for
        # each sequence.  We group losses into 1dB R bins, kind of like a histogram.
        R_min = int(np.ceil(np.min(R_loss[:,0])))
        R_max = int(np.ceil(np.max(R_loss[:,0])))
        R_mean_loss = np.zeros((R_max-R_min,2))
        # group the losses from training into 1dB wide bins, and find mean for that bin
        r = np.arange(R_min,R_max)
        for i in np.arange(len(r)):
            R = r[i]
            x = np.where(np.abs(R_loss[:,0] - R) < 0.5)
            R_mean_loss[i,0] = R
            R_mean_loss[i,1] = np.mean(R_loss[x,1])
        plt.figure(2)
        plt.plot(R_mean_loss[:,0],R_mean_loss[:,1],'b+-')
        plt.grid()
        plt.xlabel('R (dBm)')
        plt.ylabel('Loss')
        plt.show(block=False)
        plt.savefig(args.plot_R + '_loss_RdBm.png')

        np.savetxt(args.plot_R + '_loss_RdBm' + '.txt', R_mean_loss)
        quit()

    # ---------------------------------------------------------------------------------------------
    # Regular training loop
    # ---------------------------------------------------------------------------------------------

    if args.plot_loss:
        plt.figure(1)
        loss_epoch=np.zeros((args.epochs+1))
    
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
                output = model(features,H)
                loss_by_batch = distortion_loss(features, output["features_hat"])
                total_loss = torch.mean(loss_by_batch)
                total_loss.backward()
                optimizer.step()
                scheduler.step()

                running_total_loss += float(total_loss.detach().cpu())
                
                if (i + 1) % log_interval == 0:
                    current_loss = (running_total_loss - previous_total_loss) / log_interval
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

