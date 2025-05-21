"""
ML frame sync experiment.

Train a binary classification model to indentify the start of an OFDM frame.
Due to mapping into OFDM frames there are two alignment possibilities.

Create z_hat training data using a RADE encoder. Uses train.py, but with an existing model in
"plot loss" mode, ie. running through the training datasets without any additional training. 
Applies a bunch of channel impairments.
  
  python3 train.py --cuda-visible-devices 0 --sequence-length 400 --batch-size 512 --epochs 200 --lr 0.003 \
    --lr-decay-factor 0.0001 ~/Downloads/tts_speech_16k_speexdsp.f32 tmp --bottleneck 3 \
    --h_file h_nc20_mpp_train.c64 --h_complex --range_EbNo --range_EbNo_start 0 --plot_loss \
    --auxdata --timing_rand --freq_rand --plot_EqNo 250506 \
    --initial-checkpoint 250506/checkpoints/checkpoint_epoch_200.pth --write_latent z_train_250506.f32

Train frame sync detector using first 1E6 vectors and write model file

  python3 ml_sync.py z_train_250506.f32 --batch_size 512 --count 1000000 --epochs 20 --save_model 250508_ml_sync
  
Test in inference mode outside of training dataset (second 1E6 vectors), write classification values to 
y_hat.f32 for plotting:

  python3 ml_sync.py z_train_250506.f32 --count 100000 --start 1000000 --inference 250508_ml_sync --write_y_hat y_hat.f32

"""

import torch
from torch import nn
import numpy as np
import argparse

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                z_hat_file,
                latent_dim=80,
                count=-1):

        self.latent_dim = latent_dim
        # training set of correctly aligned z_hat vectors
        self.z_hat = np.fromfile(z_hat_file, count=count*latent_dim, dtype=np.float32)
        self.num_vecs = 2*(self.z_hat.shape[0] // latent_dim) - 1
        print(f"loaded {self.num_vecs:d} z_hat vecs")

    def __len__(self):
        return self.num_vecs

    def __getitem__(self, index):
        # advance by half frame at a time to generate dataset of true and false cases
        half_frame = self.latent_dim//2
        features = self.z_hat[index*half_frame:index*half_frame+self.latent_dim]
        y = np.array([(index+1) % 2], dtype=np.float32)
        return features, y
   
parser = argparse.ArgumentParser()
parser.add_argument('z_hat_file', type=str, help='file of z vectors in .f32')
parser.add_argument('--count', type=int, default=-1, help='number of z vecs to load (default all)')
parser.add_argument('--latent_dim', type=int, default=80, help='dimension of z vectors')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--save_model', type=str, default="", help='filename of model to save')
parser.add_argument('--inference', type=str, default="", help='Inference only with filename of saved model (default training mode)')
parser.add_argument('--start', type=int, default=0, help='start index of z_vec in inference mode (default 0)')
parser.add_argument('--write_y_hat', type=str, default=0, help='Write binary classifier output values to .f32 file')
args = parser.parse_args()
latent_dim = args.latent_dim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using {device} device")
 
class FrameSyncNet(nn.Module):
    def __init__(self, input_dim):
        w1 = 64
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(input_dim, w1),
            nn.ReLU(),
            nn.Linear(w1, w1),
            nn.ReLU(),
            nn.Linear(w1, 1),
            nn.Sigmoid()
       )

    def forward(self, x):
        y = self.linear_stack(x)
        return y
    
model = FrameSyncNet(latent_dim).to(device)
print(model)
num_weights = sum(p.numel() for p in model.parameters())
print(f"weights: {num_weights}")

if len(args.inference) == 0:
    # training mode
    dataset = Dataset(args.z_hat_file, count=args.count)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        sum_loss = 0.0
        sum_acc = 0.0
        for batch,(f,y) in enumerate(dataloader):
            f = f.to(device)
            y = y.to(device)
            y_hat = model(f)
            loss = loss_fn(y_hat,y)
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            if np.isnan(loss.item()):
                print("NAN encountered - quitting (try reducing lr)!")
                quit()
            sum_loss += loss.item()
            sum_acc += (y_hat.round() == y).float().mean()

        print(f'Epoch: {epoch + 1:5d} Batches: {batch + 1:3d} Loss: {sum_loss / (batch + 1):.10f} acc: {sum_acc / (batch + 1):.10f}')

    if len(args.save_model):
        print(f"Saving model to: {args.save_model}")
        torch.save(model.state_dict(), args.save_model)
else:
    # inference using pre-trained model
    print(f"Loading model from: {args.inference}")
    model.load_state_dict(torch.load(args.inference,weights_only=True))
    model.eval()
    dataset = Dataset(args.z_hat_file, count=args.count+args.start)
    sum_acc = 0.0
    if args.count == -1:
        count = dataset.__len__()
    else:
        count = args.count
    y_hat_log = np.zeros(count, dtype=np.float32)
    with torch.no_grad():
        for i in range(args.start,args.start+count):
            (f,y) = dataset.__getitem__(i)
            f = torch.from_numpy(f).to(device)
            y_hat = model(f).cpu().detach().numpy()
            y_hat_log[i-args.start] = y_hat[0]
            sum_acc += (y_hat.round() == y[0])
    print(f"acc: {sum_acc[0]/count:f}")
    if args.write_y_hat:
        y_hat_log.tofile(args.write_y_hat)


