"""
Training the neural pitch estimator

"""

import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='.f32 sequences of corr vectors (Ry) for training')
parser.add_argument('delta', type=str, help='.f32 ground truth fine timing (delta) for training')
parser.add_argument('--gpu_index', type=int, help='GPU index to use if multiple GPUs',default = 0,required = False)
parser.add_argument('--sequence_length', type=int, help='Sequence length during training',default = 50,required = False)
parser.add_argument('--xcorr_dimension', type=int, help='Dimension of Input cross-correlation',default = 160,required = False)
parser.add_argument('--gru_dim', type=int, help='GRU Dimension',default = 64,required = False)
parser.add_argument('--output_dim', type=int, help='Output dimension',default = 160,required = False)
parser.add_argument('--learning_rate', type=float, help='Learning Rate',default = 1.0e-3,required = False)
parser.add_argument('--epochs', type=int, help='Number of training epochs',default = 50,required = False)
parser.add_argument('--choice_cel', type=str, help='Choice of Cross Entropy Loss (default or robust)',choices=['default','robust'],default = 'default',required = False)
parser.add_argument('--prefix', type=str, help="prefix for model export, default: model", default='model')
parser.add_argument('--initial-checkpoint', type=str, help='initial checkpoint to start training from, default: None', default=None)
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model (default training mode)')
parser.add_argument('--fte_ml', type=str, help='optional file to save fine time errors from ML')
parser.add_argument('--fte_dsp', type=str, help='optional file to save fine time errors from clasical DSP argmax(Ry)')
parser.add_argument('--Ncp', type=int, default=32, help='length of cyclic prefix in samples, used as outlier threshold (default 32)')

args = parser.parse_args()

# Fixing the seeds for reproducability
import time
np_seed = int(time.time())
torch_seed = int(time.time())

import torch
torch.manual_seed(torch_seed)
import numpy as np
np.random.seed(np_seed)
import tqdm
from models_ft import ftDNNXcorr, ftDNNDataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ft_nn = ftDNNXcorr(args.xcorr_dimension, args.gru_dim, args.output_dim)

if type(args.initial_checkpoint) != type(None):
    checkpoint = torch.load(args.initial_checkpoint, map_location='cpu')
    ft_nn.load_state_dict(checkpoint['state_dict'], strict=False)

dataset = ftDNNDataloader(args.features,args.delta,args.output_dim,args.sequence_length)

def loss_custom(logits_softmax,labels,choice = 'default',nmax = 192,q = 0.7):
    labels_one_hot = torch.nn.functional.one_hot(labels.long(),nmax)

    if choice == 'default':
        # Categorical Cross Entropy
        CE = -torch.sum(torch.log(logits_softmax*labels_one_hot + 1.0e-6)*labels_one_hot,dim=-1)
        CE = torch.mean(CE)

    else:
        # Robust Cross Entropy
        CE = (1.0/q)*(1 - torch.sum(torch.pow(logits_softmax*labels_one_hot + 1.0e-7,q),dim=-1) )
        CE = torch.sum(CE)

    return CE

# for timing estimation in OFDM we don't need to have perfect timing, so measure
# accuracy in terms of std dev of timing est error in samples
def calc_ft_error_ml(delta_hat,labels,nmax = 192):
    #logits_softmax = torch.nn.Softmax(dim = 1)(logits).permute(0,2,1)
    #delta_hat = torch.argmax(logits_softmax, 2)
    # timing est is modulo nmax, e.g. for nmax=160
    # delta delta_hat error
    # 0     159       -1
    # 159   0         +1
    delta = labels.long()
    ft_error = ((delta_hat - delta + nmax/2) % nmax) - nmax/2
    ft_error = ft_error*1.
    return ft_error

# peak picking of Ry as a control
def calc_ft_error_dsp(xi,labels,nmax = 192):
    delta_hat = torch.argmax(xi, 2)
    # timing est is modulo nmax, e.g. for nmax=160
    # delta delta_hat error
    # 0     159       -1
    # 159   0         +1
    delta = labels.long()
    ft_error = ((delta_hat - delta + nmax/2) % nmax) - nmax/2
    ft_error = ft_error*1.
    return ft_error

batch_size = 256
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

ft_nn = ft_nn.to(device)
#print(ft_nn)
num_weights = sum(p.numel() for p in ft_nn.parameters())
print(f"weights: {num_weights}")

learning_rate = args.learning_rate
model_opt = torch.optim.Adam(ft_nn.parameters(), lr = learning_rate)

num_epochs = args.epochs

if len(args.inference) == 0:
    # training mode
    for epoch in range(num_epochs):
        # not we average error variances, as can't average std dev directly
        losses = []
        vars_ml = []
        vars_dsp = []
        ft_nn.train()
        with tqdm.tqdm(dataloader) as train_epoch:
            for i, (Ry, delta) in enumerate(train_epoch):
                delta, Ry = delta.to(device, non_blocking=True), Ry.to(device, non_blocking=True)
                logits_softmax = ft_nn(Ry)
                loss = loss_custom(logits_softmax = logits_softmax,labels = delta,choice = args.choice_cel,nmax = args.output_dim)
                delta_hat = torch.argmax(logits_softmax, 2)
                ft_error_ml = calc_ft_error_ml(delta_hat = delta_hat,labels = delta, nmax = args.output_dim)
                var_ml = torch.var(ft_error_ml).detach()

                ft_error_dsp = calc_ft_error_dsp(xi = Ry,labels = delta, nmax = args.output_dim)
                var_dsp = torch.var(ft_error_dsp).detach()

                model_opt.zero_grad()
                loss.backward()
                model_opt.step()

                losses.append(loss.item())
                vars_ml.append(var_ml.item())
                vars_dsp.append(var_dsp.item())
                avg_loss = np.mean(losses)
                std_ml = np.mean(vars_ml)**0.5
                std_dsp = np.mean(vars_dsp)**0.5
                train_epoch.set_postfix({"Epoch" : epoch, "loss":avg_loss, "std_ml" : std_ml, "std_dsp" : std_dsp})
    
    if len(args.save_model):
        print(f"Saving model to: {args.save_model}")
        torch.save(ft_nn.state_dict(), args.save_model)

else:
    # inference using pre-trained model
    print(f"Loading model from: {args.inference}")
    ft_nn.load_state_dict(torch.load(args.inference,weights_only=True))
    ft_nn.eval()

    if args.fte_ml:
        f_fte_ml = open(args.fte_ml,"wb")
    if args.fte_dsp:
        f_fte_dsp = open(args.fte_dsp,"wb")

    ft_errors_ml = []
    ft_errors_dsp = []
    for i in range(dataset.__len__()):
        Ry, delta = dataset.__getitem__(i)
        Ry = torch.reshape(Ry,(1,Ry.shape[0],Ry.shape[1]))
        delta = torch.reshape(delta,(1,-1))
        delta, Ry = delta.to(device, non_blocking=True), Ry.to(device, non_blocking=True)
        logits_softmax = ft_nn(Ry)
        delta_hat = torch.argmax(logits_softmax, 2)
        ft_error_ml = calc_ft_error_ml(delta_hat = delta_hat,labels = delta, nmax = args.output_dim)
        ft_errors_ml.append(ft_error_ml.detach().cpu().flatten())
        if args.fte_ml:
            ft_error_ml.cpu().detach().numpy().flatten().astype('float32').tofile(f_fte_ml)

        ft_error_dsp = calc_ft_error_dsp(xi = Ry,labels = delta, nmax = args.output_dim)
        ft_errors_dsp.append(ft_error_dsp.detach().cpu().flatten())
        if args.fte_dsp:
            ft_error_dsp.cpu().detach().numpy().flatten().astype('float32').tofile(f_fte_dsp)

    std_ml = np.std(ft_errors_ml)
    mean_ml = np.mean(ft_errors_ml)
    Noutliers_ml = float(len(np.argwhere(np.abs(ft_errors_ml-mean_ml) > args.Ncp/2)))
    std_dsp = np.std(ft_errors_dsp)
    Noutliers_dsp = float(len(np.argwhere(np.abs(ft_errors_dsp) > args.Ncp/2)))
    N = dataset.__len__()*args.sequence_length
    outliers_ml = 100*Noutliers_ml/N
    outliers_dsp = 100*Noutliers_dsp/N
    print(f"N: {dataset.__len__()*args.sequence_length:d} std_ml: {std_ml:5.2f} outliers_ml: {outliers_ml:5.3f}% std_dsp: {std_dsp:5.2f} outliers_dsp: {outliers_dsp:5.3f}%")
    
    if args.fte_ml:
        f_fte_ml.close()
    if args.fte_dsp:
        f_fte_dsp.close()

