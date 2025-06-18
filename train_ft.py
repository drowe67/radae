"""
Training the neural pitch estimator

"""

import os
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('features', type=str, help='.f32 sequences of corr vectors (Ry) for training')
parser.add_argument('delta', type=str, help='.f32 ground truth fine timing (delta) for training')
parser.add_argument('output_folder', type=str, help='Output directory to store the model weights and config')
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

args = parser.parse_args()

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

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


dataset_training = ftDNNDataloader(args.features,args.delta,args.output_dim,args.sequence_length)

def loss_custom(logits,labels,choice = 'default',nmax = 192,q = 0.7):
    logits_softmax = torch.nn.Softmax(dim = 1)(logits).permute(0,2,1)
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
def accuracy(logits,labels,nmax = 192):
    logits_softmax = torch.nn.Softmax(dim = 1)(logits).permute(0,2,1)
    delta_hat = torch.argmax(logits_softmax, 2)
    # timing est is modulo nmax, e.g. for nmax=160
    # delta delta_hat error
    # 0     159       -1
    # 159   0         +1
    delta = labels.long()
    ft_error = ((delta_hat - delta + nmax/2) % nmax) - nmax/2
    ft_error = ft_error*1.
    return torch.std(ft_error)

# peak picking of Ry as a control
def accuracy_xi(xi,labels,nmax = 192):
    delta_hat = torch.argmax(xi, 2)
    # timing est is modulo nmax, e.g. for nmax=160
    # delta delta_hat error
    # 0     159       -1
    # 159   0         +1
    delta = labels.long()
    ft_error = ((delta_hat - delta + nmax/2) % nmax) - nmax/2
    ft_error = ft_error*1.
    return torch.std(ft_error)

train_dataset, test_dataset = torch.utils.data.random_split(dataset_training, [0.95,0.05], generator=torch.Generator().manual_seed(torch_seed))

batch_size = 256
train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

ft_nn = ft_nn.to(device)
#num_params = count_parameters(ft_nn)
learning_rate = args.learning_rate
model_opt = torch.optim.Adam(ft_nn.parameters(), lr = learning_rate)

num_epochs = args.epochs

for epoch in range(num_epochs):
    losses = []
    accs = []
    accs_xi = []
    ft_nn.train()
    with tqdm.tqdm(train_dataloader) as train_epoch:
        for i, (Ry, delta) in enumerate(train_epoch):
            delta, Ry = delta.to(device, non_blocking=True), Ry.to(device, non_blocking=True)
            delta_hat = ft_nn(Ry)
            loss = loss_custom(logits = delta_hat,labels = delta,choice = args.choice_cel,nmax = args.output_dim)
            acc = accuracy(logits = delta_hat,labels = delta, nmax = args.output_dim)
            acc = acc.detach()
            acc_xi = accuracy_xi(xi = Ry,labels = delta, nmax = args.output_dim)
            acc_xi = acc_xi.detach()

            model_opt.zero_grad()
            loss.backward()
            model_opt.step()

            losses.append(loss.item())
            accs.append(acc.item())
            accs_xi.append(acc_xi.item())
            avg_loss = np.mean(losses)
            avg_acc = np.mean(accs)
            avg_acc_xi = np.mean(accs_xi)
            train_epoch.set_postfix({"Train Epoch" : epoch, "Train Loss":avg_loss, "acc" : avg_acc.item(),  "acc_xi" : avg_acc_xi.item()})

    if epoch % 10 == 0:
        ft_nn.eval()
        losses = []
        accs = []
        with tqdm.tqdm(test_dataloader) as test_epoch:
            for i, (Ry, delta) in enumerate(test_epoch):
                delta, Ry = delta.to(device, non_blocking=True), Ry.to(device, non_blocking=True)
                delta_hat = ft_nn(Ry)
                
                loss = loss_custom(logits = delta_hat,labels = delta,choice = args.choice_cel,nmax = args.output_dim)
                acc = accuracy(logits = delta_hat,labels = delta,nmax = args.output_dim)
                acc = acc.detach()
                losses.append(loss.item())
                accs.append(acc.item())
                avg_loss = np.mean(losses)
                avg_acc = np.mean(accs)
                test_epoch.set_postfix({"Epoch" : epoch, "Test Loss":avg_loss, "Test acc" : avg_acc.item()})

ft_nn.eval()

config = dict(
    epochs=num_epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    np_seed=np_seed,
    torch_seed=torch_seed,
    xcorr_dim=args.xcorr_dimension,
    gru_dim=args.gru_dim,
    output_dim=args.output_dim,
    choice_cel=args.choice_cel,
    sequence_length=args.sequence_length,
)

#model_save_path = os.path.join(args.output_folder, f"{args.prefix}.pth")
#print(model_save_path)
#checkpoint = {
#    'state_dict': pitch_nn.state_dict(),
#    'config': config
#}
#torch.save(checkpoint, model_save_path)
