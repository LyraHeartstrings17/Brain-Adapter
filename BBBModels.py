import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from sc_mbm.mae_for_eeg import PatchEmbed1D
from timm.models.vision_transformer import Block
import sc_mbm.utils as ut
from torch import nn


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
        self.output = nn.Linear(d_lstm_size, output_size)
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
        # d_output, d_hidden_state = self.decoder(e_output[0])
        x = e_output[0][:, -1, :]
        reps = x
        # Forward output
        xa = F.relu(self.output(x))
        x = self.classifier(xa)
        return x, xa

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        return


class TimeFreqEncoder(nn.Module):
    def __init__(self, pretrained_model_time, pretrained_model_freq):
        super(TimeFreqEncoder, self).__init__()

        self.pretrained_model_time = pretrained_model_time
        self.pretrained_model_time.nocliptune = True
        self.pretrained_model_time.linear_proba = False
        self.pretrained_model_freq = pretrained_model_freq

        # self.fc01 = nn.Linear(1024 + 256, 40)

    def forward(self, x):
        time_feature = self.pretrained_model_time(x)
        lstmcls, freq_feature = self.pretrained_model_freq(x)
        x = torch.cat((time_feature, freq_feature), dim=1)

        lastrep = x
        encoded = x
        x = self.fc01(encoded)
        #
        scores = x
        return lastrep, encoded, scores


class AlignNet(nn.Module):
    def __init__(self, input_size, freq_size, output_size, pretrained_model):
        super(AlignNet, self).__init__()

        self.pretrained_model = pretrained_model  #TimeFreqEncoder
        # for param in self.pretrained_model.parameters():
        #     param.requires_grad = False

        self.fc01 = nn.Linear(input_size + freq_size, 4 * input_size)
        self.fc02 = nn.Linear(4 * input_size, input_size)
        self.fc03 = nn.Linear(input_size, 4 * input_size)
        self.fc04 = nn.Linear(4 * input_size, input_size)
        self.fc05 = nn.Linear(input_size, 4 * input_size)
        self.fc06 = nn.Linear(4 * input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lastrep, encoded = self.pretrained_model(x)
        x = lastrep
        x = self.fc01(x)
        x = self.relu(x)
        res_4is_1 = x
        x = self.fc02(x)
        x = self.relu(x)
        res_is_2 = x
        x = self.fc03(x) + res_4is_1
        x = self.relu(x)
        res_4is_2 = x
        x = self.fc04(x) + res_is_2
        x = self.relu(x)
        x = self.fc05(x) + res_4is_2
        x = self.relu(x)
        x = self.fc06(x)
        return x


class AlignNet_OnlyT(nn.Module):
    def __init__(self, input_size, output_size, pretrained_model):
        super(AlignNet_OnlyT, self).__init__()

        self.pretrained_model = pretrained_model  # TimeEncoder
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.fc01 = nn.Linear(input_size, 4 * input_size)
        self.fc02 = nn.Linear(4 * input_size, input_size)
        self.fc03 = nn.Linear(input_size, 4 * input_size)
        self.fc04 = nn.Linear(4 * input_size, input_size)
        self.fc05 = nn.Linear(input_size, 4 * input_size)
        self.fc06 = nn.Linear(4 * input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        time_feature = self.pretrained_model(x)
        x = time_feature
        x = self.fc01(x)
        x = self.relu(x)
        res_4is_1 = x
        x = self.fc02(x)
        x = self.relu(x)
        res_is_2 = x
        x = self.fc03(x) + res_4is_1
        x = self.relu(x)
        res_4is_2 = x
        x = self.fc04(x) + res_is_2
        x = self.relu(x)
        x = self.fc05(x) + res_4is_2
        x = self.relu(x)
        x = self.fc06(x)
        return x


class AlignNet_OnlyF(nn.Module):
    def __init__(self, input_size, output_size, pretrained_model):
        super(AlignNet_OnlyF, self).__init__()

        self.pretrained_model = pretrained_model  # FreqEncoder
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        self.fc01 = nn.Linear(input_size, 4 * input_size)
        self.fc02 = nn.Linear(4 * input_size, input_size)
        self.fc03 = nn.Linear(input_size, 4 * input_size)
        self.fc04 = nn.Linear(4 * input_size, input_size)
        self.fc05 = nn.Linear(input_size, 4 * input_size)
        self.fc06 = nn.Linear(4 * input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstmcls, freq_feature = self.pretrained_model(x)
        x = freq_feature
        x = self.fc01(x)
        x = self.relu(x)
        res_4is_1 = x
        x = self.fc02(x)
        x = self.relu(x)
        res_is_2 = x
        x = self.fc03(x) + res_4is_1
        x = self.relu(x)
        res_4is_2 = x
        x = self.fc04(x) + res_is_2
        x = self.relu(x)
        x = self.fc05(x) + res_4is_2
        x = self.relu(x)
        x = self.fc06(x)
        return x


class TimeEncoder(nn.Module):
    def __init__(self, time_len=512, patch_size=4, embed_dim=1024, in_chans=128,
                 depth=24, num_heads=16, mlp_ratio=1., norm_layer=nn.LayerNorm, global_pool=True):
        super().__init__()
        self.patch_embed = PatchEmbed1D(time_len, patch_size, in_chans, embed_dim)

        num_patches = int(time_len / patch_size)

        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        # print(x.shape)
        # print(self.pos_embed[:, 1:, :].shape)
        x = x + self.pos_embed[:, 1:, :]
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        # print(x.shape)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        # print(x.shape)
        x = self.norm(x)
        # print(x.shape)
        return x

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs)  # N, n_seq, embed_dim
        return latent.squeeze(1)  # N, embed_dim

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        return


class classify_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(19, 1, 1, stride=1)  # nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(1024, 40)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class mapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.Conv1d(19, 1, 1, stride=1)  # nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(1024, 768)

    def forward(self, x):
        x = self.maxpool(x)
        x = x.squeeze(1)
        x = self.fc(x)
        return x


class FreqEncoder(nn.Module):

    def __init__(self, d_lstm_size=64, output_size=256, e_lstm_model=None, d_lstm_model=None):
        # Call parent
        super().__init__()
        # Define parameters
        self.lstm_size = d_lstm_size
        self.output_size = output_size

        # Define internal modules
        self.lstm = e_lstm_model
        self.decoder = d_lstm_model
        self.output = nn.Linear(d_lstm_size, output_size)
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

        e_output, hidden_state = self.lstm(x)
        # d_output, d_hidden_state = self.decoder(e_output[0])
        x = e_output[0][:, -1, :]
        reps = x
        # Forward output
        xa = F.relu(self.output(x))
        x = self.classifier(xa)
        return x, xa

    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)

        m, u = self.load_state_dict(state_dict, strict=False)
        return


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
        self.output = nn.Linear(d_lstm_size, output_size)
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
        # d_output, d_hidden_state = self.decoder(e_output[0])
        x = e_output[0][:, -1, :]
        reps = x
        # Forward output
        xa = F.relu(self.output(x))
        x = self.classifier(xa)
        return x, xa


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
                output_inner.append(torch.squeeze(h, -1))

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
