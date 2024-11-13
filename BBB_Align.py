import numpy as np
import torch
from scipy.interpolate import interp1d
from torch import nn
import argparse

from torch.utils.data import DataLoader, ConcatDataset
import sys
import config
from model.BBBModels import AlignNet, TimeEncoder, FreqEncoder, TimeFreqEncoder, ConvLSTM, ConvLSTMCell, \
    ConvFreqEncoder, AlignNet_OnlyF, AlignNet_OnlyT

import os


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class AlignLoss:
    def __init__(self):
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CosineEmbeddingLoss()

    def compute(self, rep_mask, rep_mask_prediction, neg_rep_mask_prediction):
        mse_loss = self.mse(rep_mask, rep_mask_prediction)
        target_labels = torch.ones(len(rep_mask_prediction)).to("cuda")  
        pos_loss = self.ce(rep_mask_prediction, rep_mask, target_labels)
        contrastive_loss = pos_loss
        return mse_loss, contrastive_loss


class AlignDataset:
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
        # get ip_emb
        image_name = self.images[self.data[i]["image"]]
        pos_emb = np.loadtxt('./data/img_ip_emb/' + image_name + 'prompt_embeds' + '.csv', delimiter=',')
        neg_emb = np.loadtxt('./data/img_ip_emb_neg/' + image_name + 'negative_prompt_embeds' + '.csv', delimiter=',')
        # Return

        return eeg, label, torch.tensor(pos_emb).reshape(1, -1).squeeze(), torch.tensor(neg_emb).reshape(1,
                                                                                                         -1).squeeze()


class Splitter:

    def __init__(self, dataset, split_path, split_num=0, split_name="", subject=4):
        # Set EEG dataset
        self.dataset = dataset
        # Load split
        loaded = torch.load(split_path)

        self.split_idx = loaded["splits"][split_num][split_name]
        # Filter data
        self.split_idx = [i for i in self.split_idx if
                          i <= len(self.dataset.data) and 450 <= self.dataset.data[i]["eeg"].size(1) <= 600]
        # Compute size

        self.size = len(self.split_idx)
        self.num_voxels = 440
        self.data_len = 512

    # Get size
    def __len__(self):
        return self.size

    # Get item
    def __getitem__(self, i):
        return self.dataset[self.split_idx[i]]


def main():
    eval_cosine = 1.0
    dataset = AlignDataset(eeg_signals_path='',
                           subject=0)
    train_dataset = Splitter(dataset,
                             split_path='',
                             split_num=0, split_name="train")
    val_dataset = Splitter(dataset,
                           split_path='',
                           split_num=0, split_name="val")
    test_dataset = Splitter(dataset,
                            split_path='',
                            split_num=0, split_name="test")
    train_dataset = ConcatDataset([train_dataset, val_dataset])
    dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    print(len(train_dataset))
    time_enc = TimeEncoder()
    metafile = torch.load(
        "",
        map_location='cpu')
    time_enc.load_checkpoint(metafile['model'])
    time_enc.to('cuda')

    freq_enc = torch.load(
        "",
        map_location='cpu')
    freq_enc.to('cuda')
    tf_enc = TimeFreqEncoder(time_enc, freq_enc)
    tf_enc.to('cuda')
    alignmodel = AlignNet_OnlyF(256, 4 * 768, pretrained_model=freq_enc)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        alignmodel = nn.DataParallel(alignmodel)
    alignmodel.to('cuda')
    print('IP_finetune only F')
    optimizer = torch.optim.AdamW(alignmodel.parameters(), lr=0.0005)
    loss_function = AlignLoss()
    for epoch in range(500):
        print('Epoch:' + str(epoch + 1))
        alignmodel.train()
        loss_mse = 0
        loss_contrastive = 0
        loss_sum = 0

        teloss_mse = 0
        teloss_contrastive = 0
        teloss_sum = 0

        for idx, batch in enumerate(dataloader):
            batch = [x.to('cuda').to(torch.float32) for x in batch]
            optimizer.zero_grad()

            ip_pred = alignmodel.forward(batch[0])
            mse_loss, contrastive_loss = loss_function.compute(ip_pred, batch[2], batch[3])
            loss_contrastive += contrastive_loss.item()
            loss_mse += mse_loss.item()
            loss = contrastive_loss  # +l1_penalty
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        for name, param in alignmodel.named_parameters():
            if param.grad is not None:
                print(f"Gradient of {name}: {param.grad}")

        trloss = loss_sum / (idx + 1)
        trmse = loss_mse / (idx + 1)
        trcosine = loss_contrastive / (idx + 1)

        for idxte, batch in enumerate(test_loader):
            alignmodel.eval()
            batch = [x.to('cuda').to(torch.float32) for x in batch]
            ip_pred = alignmodel.forward(batch[0])
            mse_loss, contrastive_loss = loss_function.compute(ip_pred, batch[2], batch[3])
            t_loss = contrastive_loss
            teloss_contrastive += contrastive_loss.item()
            teloss_mse += mse_loss.item()
            teloss_sum += t_loss.item()

        teloss = teloss_sum / (idxte + 1)
        temse = teloss_mse / (idxte + 1)
        tecosine = teloss_contrastive / (idxte + 1)

        print(
            'ip_finetune epoch{0}, trloss{1}, trmse{2}, trcosine{3}, teloss{4}, temse{5}, tecosine{6}'.format(epoch + 1,
                                                                                                              trloss,
                                                                                                              trmse,
                                                                                                              trcosine,
                                                                                                              teloss,
                                                                                                              temse,
                                                                                                              tecosine))

        if (epoch + 1) % 100 == 0:
            torch.save(alignmodel.state_dict(),
                       '' + str(
                           epoch + 1) + 'frozeT' + '.pkl')# AblationOnlyT

        if tecosine < eval_cosine:
            eval_cosine = tecosine
            torch.save(alignmodel.state_dict(),
                       '')




main()
