import os
import random

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
import torch.nn.functional as F


def fft(x):
    x = x.cpu()
    fourier_transform = np.fft.fft(x)
    half_spectrum = fourier_transform[:, :, 1:440 // 2 + 1]
    amplitude_spectrum = np.abs(half_spectrum)

    amplitude_spectrum = torch.tensor(amplitude_spectrum).float()

    x = amplitude_spectrum
    x = x.to("cuda")
    return x


class ConvFreqEncoder(nn.Module):

    def __init__(self, d_lstm_size=64, output_size=256, e_lstm_model=None, d_lstm_model=None):
        # Call parent
        super().__init__()
        # Define parameters
        self.lstm_size = d_lstm_size
        self.output_size = output_size

        # Define internal modules
        self.encoder = e_lstm_model
        self.decoder = d_lstm_model
        self.relu = nn.ReLU()
        self.output = nn.Linear(d_lstm_size, output_size)
        self.softmax = nn.Softmax
        self.classifier = nn.Linear(output_size, 40)

    def forward(self, x):
        x = x.cpu()
        fourier_transform = np.fft.fft(x)
        half_spectrum = fourier_transform[:, :, 1:440 // 2 + 1]
        amplitude_spectrum = np.abs(half_spectrum)

        amplitude_spectrum = torch.tensor(amplitude_spectrum).float()

        x = amplitude_spectrum
        x = x.to("cuda")

        # lstm_init = (torch.zeros(self.lstm_layers, batch_size, self.lstm_size),
        #              torch.zeros(self.lstm_layers, batch_size, self.lstm_size))
        # if x.is_cuda: lstm_init = (lstm_init[0].cuda(), lstm_init[0].cuda())
        # lstm_init = (Variable(lstm_init[0], volatile=x.volatile), Variable(lstm_init[1], volatile=x.volatile))

        e_output, hidden_state = self.encoder(x)
        if self.decoder is None:
            # x = e_output[0][:, -1, :]
            # # Forward output
            # xa = F.relu(self.output(x))
            # cls = self.classifier(xa)
            # return cls, xa
            x = e_output[0][:, -1, :]
            # Forward output
            xa = F.relu(self.output(x))
            cls = self.classifier(xa)
            return cls, xa
        else:
            d_output, d_hidden_state = self.decoder(e_output[0])
            return d_output, d_hidden_state


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        self.conv = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        input_tensor = torch.unsqueeze(input_tensor, 2)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_dim, 1).to(self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, 1).to(self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, bias=True,
                 return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2)

        batch_size, seq_len, _ = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size)
        output_inner = []
        layer_output_list = []
        last_state_list = []
        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :],
                                                 cur_state=[h, c])
                a = torch.squeeze(h)
                output_inner.append(torch.squeeze(h))

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    # for t in range(seq_len):
    #     input_at_time = input_tensor[:, t, :]
    #     output_at_time = []
    #     for i, cell in enumerate(self.cell_list):
    #         h, c = hidden_state[i]
    #         output_at_time.append(cell(input_at_time, (h, c)))
    #         hidden_state[i] = output_at_time[-1]
    #     output_inner.append(torch.stack(output_at_time, dim=0))
    #
    # output = torch.stack(output_inner, dim=1)
    # if not self.return_all_layers:
    #     output = output[-1]
    #     hidden_state = hidden_state[-1]
    #
    # return output, hidden_state
    def _init_hidden(self, batch_size):
        return [self.cell_list[i].init_hidden(batch_size) for i in range(len(self.cell_list))]


class EEGDataset:

    # Constructor
    def __init__(self, eeg_signals_path, subject):
        # Load EEG signals
        loaded = torch.load(eeg_signals_path)
        if subject != 0:
            self.data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                         loaded['dataset'][i]['subject'] == subject]
        else:
            self.data = loaded['dataset']
        self.labels = loaded["labels"]
        self.images = loaded["images"]
        # Compute size
        self.size = len(self.data)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Process EEG
        eeg = self.data[i]["eeg"].float()

        eeg = eeg[:, 20:460]
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, 512)
        f = interp1d(x, eeg)
        eeg = torch.tensor(f(x2))
        eeg = eeg.transpose(0, 1)
        # p1 = eeg[:, 22:32]
        # p2 = eeg[:, 55:64] * 5
        # eeg = torch.cat((p1, p2), dim=1)
        eeg = eeg.transpose(0, 1)
        label = self.data[i]["label"]
        while True:
            j = random.randint(0, 1000)
            if self.data[j]["label"] != label:
                break
        # Return

        return eeg, label


# Splitter class
class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="train"):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)
        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size
        self.size = len(self.split_idx)

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        # Get sample from dataset
        eeg, label, img_emb, neg_img_emb = self.dataset[self.split_idx[i]]
        # Return
        return eeg, label, img_emb, neg_img_emb


# Load dataset
dataset = EEGDataset('', 0)
# Create loaders
loaders = {split: DataLoader(
    Splitter(dataset, split_path='',
             split_num=0, split_name=split),
    batch_size=128, drop_last=True, shuffle=True) for split in
    ["train", "val", "test"]}
train_dataset = Splitter(dataset,
                         split_path='',
                         split_num=0, split_name="train")
print(len(train_dataset))

# Load model

# Create discriminator model/optimizer
input_dim = 220  # 特征维度
hidden_dim = [64, 96, 128]  # 隐藏层维度
kernel_size = [3, 3, 3]  # 卷积核大小
num_layers = 3  # LSTM层数
# 创建模型
encoder = ConvLSTM(input_dim, hidden_dim, kernel_size, num_layers).to('cuda')
model = ConvFreqEncoder(d_lstm_size=hidden_dim[-1], e_lstm_model=encoder, output_size=128)
model = model.to('cuda')
optimizer = getattr(torch.optim, 'Adam')(model.parameters(), lr=0.001)

losses_per_epoch = {"train": [], "val": [], "test": []}
accuracies_per_epoch = {"train": [], "val": [], "test": []}

best_accuracy = 0
best_accuracy_val = 0
best_epoch = 0
# Start training

predicted_labels = []
correct_labels = []

for epoch in range(1, 500 + 1):
    # Initialize loss/accuracy variables
    losses = {"train": 0, "val": 0, "test": 0}
    accuracies = {"train": 0, "val": 0, "test": 0}
    counts = {"train": 0, "val": 0, "test": 0}
    # Process each split
    for split in ("train", "val", "test"):
        # Set network mode
        if split == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            model.eval()
            torch.set_grad_enabled(False)
        # Process all split batches
        for i, (input, target) in enumerate(loaders[split]):
            input = input.to("cuda")
            target = target.to("cuda")
            # Forward
            output, xa = model(input)
            # Compute loss
            loss = F.cross_entropy(output, target)
            losses[split] += loss.item()
            # Compute accuracy
            _, pred = output.data.max(1)
            correct = pred.eq(target.data).sum().item()
            accuracy = correct / input.data.size(0)
            accuracies[split] += accuracy
            counts[split] += 1
            # Backward and optimize
            if split == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    TrL, VL, TeL = losses["train"] / counts["train"], losses[
        "val"] / counts["val"], losses["test"] / counts["test"]
    TrA, VA, TeA = accuracies["train"] / counts["train"], accuracies[
        "val"] / counts["val"], accuracies["test"] / counts["test"]
    print("Epoch {0}:TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}".format(epoch, TrL, TrA,
                                                                                                        VL, VA, TeL,
                                                                                                        TeA))
    # Print info at the end of the epoch
    # if accuracies["val"] / counts["val"] >= best_accuracy_val:
    #     best_accuracy_val = accuracies["val"] / counts["val"]
    #     best_accuracy = accuracies["test"] / counts["test"]
    #     best_epoch = epoch
    #
    # TrL, TrA, VL, VA, TeL, TeA = losses["train"] / counts["train"], accuracies["train"] / counts["train"], losses[
    #     "val"] / counts["val"], accuracies["val"] / counts["val"], losses["test"] / counts["test"], accuracies["test"] / \
    #                              counts["test"]
    # print(
    #     "Model: {11} - Subject {12} - Time interval: [{9}-{10}]  [{9}-{10} Hz] - Epoch {0}: TrL={1:.4f}, TrA={2:.4f}, VL={3:.4f}, VA={4:.4f}, TeL={5:.4f}, TeA={6:.4f}, TeA at max VA = {7:.4f} at epoch {8:d}".format(
    #         epoch,
    #         losses["train"] / counts["train"],
    #         accuracies["train"] / counts["train"],
    #         losses["val"] / counts["val"],
    #         accuracies["val"] / counts["val"],
    #         losses["test"] / counts["test"],
    #         accuracies["test"] / counts["test"],
    #         best_accuracy, best_epoch, 20, 460, 'lstm', 4))
    #
    # losses_per_epoch['train'].append(TrL)
    # losses_per_epoch['val'].append(VL)
    # losses_per_epoch['test'].append(TeL)
    # accuracies_per_epoch['train'].append(TrA)
    # accuracies_per_epoch['val'].append(VA)
    # accuracies_per_epoch['test'].append(TeA)

    if epoch % 100 == 0:
        torch.save(model,
                   os.path.join('./pre_train_model/cls',
                                '%s__subject%d_epoch_%d.pth' % ('conv_lstm_4layers_cls', 0, epoch)))
