import os

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
from einops import rearrange
from ip_adapter import IPAdapter
from scipy.interpolate import interp1d
from sklearn.svm import SVC

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import transforms

from model.BBBModels import AlignNet, TimeEncoder, FreqEncoder, TimeFreqEncoder, ConvFreqEncoder, ConvLSTM, \
    ConvLSTMCell, AlignNet_OnlyF, AlignNet_OnlyT
from BBB_Align import AlignDataset, Splitter

propmt_dict = {'n02106662': 'german shepherd dog',
               'n02124075': 'egyptian cat ',
               'n02281787': 'lycaenid butterfly',
               'n02389026': 'sorrel horse',
               'n02492035': 'Cebus capucinus',
               'n02504458': 'African elephant',
               'n02510455': 'panda',
               'n02607072': 'anemone fish',
               'n02690373': 'airliner',
               'n02906734': 'broom',
               'n02951358': 'canoe or kayak',
               'n02992529': 'cellular telephone',
               'n03063599': 'coffee mug',
               'n03100240': 'old convertible',
               'n03180011': 'desktop computer',
               'n03197337': 'digital watch',
               'n03272010': 'electric guitar',
               'n03272562': 'electric locomotive',
               'n03297495': 'espresso maker',
               'n03376595': 'folding chair',
               'n03445777': 'golf ball',
               'n03452741': 'grand piano',
               'n03584829': 'smoothing iron',
               'n03590841': 'Orange jack-o’-lantern',
               'n03709823': 'mailbag',
               'n03773504': 'missile',
               'n03775071': 'mitten,glove',
               'n03792782': 'mountain bike, all-terrain bike',
               'n03792972': 'mountain tent',
               'n03877472': 'pajama',
               'n03888257': 'parachute',
               'n03982430': 'pool table, billiard table, snooker table ',
               'n04044716': 'radio telescope',
               'n04069434': 'eflex camera',
               'n04086273': 'revolver, six-shooter',
               'n04120489': 'running shoe',
               'n07753592': 'banana',
               'n07873807': 'pizza',
               'n11939491': 'daisy',
               'n13054560': 'bolete'
               }

lable_number_dict = {
    '12': 'n02106662',
    '39': 'n02124075',
    '11': 'n02281787',
    '0': 'n02389026',
    '21': 'n02492035',
    '35': 'n02504458',
    '8': 'n02510455',
    '3': 'n02607072',
    '36': 'n02690373',
    '18': 'n02906734',
    '10': 'n02951358',
    '15': 'n02992529',
    '5': 'n03063599',
    '24': 'n03100240',
    '17': 'n03180011',
    '34': 'n03197337',
    '28': 'n03272010',
    '37': 'n03272562',
    '4': 'n03297495',
    '25': 'n03376595',
    '16': 'n03445777',
    '30': 'n03452741',
    '2': 'n03584829',
    '14': 'n03590841',
    '23': 'n03709823',
    '20': 'n03773504',
    '27': 'n03775071',
    '6': 'n03792782',
    '31': 'n03792972',
    '26': 'n03877472',
    '1': 'n03888257',
    '22': 'n03982430',
    '38': 'n04044716',
    '29': 'n04069434',
    '7': 'n04086273',
    '13': 'n04120489',
    '32': 'n07753592',
    '19': 'n07873807',
    '9': 'n11939491',
    '33': 'n13054560'
}


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0  # to -1 ~ 1
    return img


def channel_last(img):
    if img.shape[-1] == 3:
        return img
    return rearrange(img, 'c h w -> h w c')


def main():
    loaded = torch.load('/datasets/eeg_5_95_std.pth')
    imagenet = '/datasets/imageNet_images'
    loaded2 = torch.load('/datasets/block_splits_by_image_single.pth')
    split_idx = loaded2["splits"][0]['test']
    time_enc = TimeEncoder()
    metafile = torch.load(
        "/results/eeg_pretrain/26-03-2024-22-04-35/checkpoints/checkpoint.pth",
        map_location='cpu')
    time_enc.load_checkpoint(metafile['model'])
    time_enc.to('cuda')

    freq_enc = torch.load("./pre_train_model/conv_lstm_4layers__subject0_epoch_1000.pth", map_location='cpu')
    freq_enc.to('cuda')
    tf_enc = TimeFreqEncoder(time_enc, freq_enc)
    tf_enc.to('cuda')
    # alignmodel = AlignNet(1024, 256, 4 * 768, pretrained_model=tf_enc)
    # align_dict = torch.load('./eval/model/500convlstm4layers_128ch.pkl', map_location='cpu')
    alignmodel = AlignNet_OnlyF(256, 4 * 768, pretrained_model=freq_enc)
    align_dict = torch.load('./eval/model/500frozeF.pkl', map_location='cpu')
    align_dict = {k.replace('module.', ''): v for k, v in align_dict.items()}
    alignmodel.load_state_dict(align_dict)
    alignmodel.to('cuda')
    alignmodel.eval()
    for k in range(6):
        pres = []
        prompts = []
        subject = k + 1
        data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
                loaded['dataset'][i]['subject'] == subject]
        images = loaded["images"]
        for i in range(len(split_idx)):
            eeg = data[split_idx[i]]['eeg']
            image_name = images[data[split_idx[i]]["image"]]
            prompt = propmt_dict[image_name.split('_')[0]]
            prompts.append(prompt)
            # image_path = os.path.join(imagenet, image_name.split('_')[0], image_name + '.JPEG')
            # image_raw = Image.open(image_path).convert('RGB')
            # image_raw.save("./eval/img/4layers-all/gt/val_" + str(i) + '_' + 'gt' + '.jpg')
            eeg = eeg[:, 20:460]
            x = np.linspace(0, 1, eeg.shape[-1])
            x2 = np.linspace(0, 1, 512)
            f = interp1d(x, eeg)
            eeg = torch.tensor(f(x2))
            ip_pred = alignmodel.forward(eeg.to('cuda').to(torch.float32).reshape(1, 128, 512)).reshape(1, 4,
                                                                                                        768).detach().to(
                'cpu')
            pres.append(ip_pred)
            # print(i)
        torch.save(pres, "eval/img/4layers-sbj" + str(subject) + "/pres")


def gen_prompt():
    loaded = torch.load('/datasets/eeg_5_95_std.pth')
    subject = 1
    data = [loaded['dataset'][i] for i in range(len(loaded['dataset'])) if
            loaded['dataset'][i]['subject'] == subject]
    loaded = torch.load('/datasets/block_splits_by_image_single.pth')
    split_idx = loaded["splits"][0]['test']
    timefreq_model = torch.load('./pre_train_model/best_time_freq_cls_model.pth')
    print('eval cls model')
    timefreq_model = timefreq_model.to('cuda')
    timefreq_model.eval()
    pres = []
    prompts = []
    paths = []
    train_pres = []
    train_paths = []
    for i in range(len(split_idx)):
        eeg = data[split_idx[i]]['eeg']
        # train_eeg = data[train_idx[i]]['eeg']
        eeg = eeg[:, 20:460]
        x = np.linspace(0, 1, eeg.shape[-1])
        x2 = np.linspace(0, 1, 512)
        f = interp1d(x, eeg)
        eeg = torch.tensor(f(x2))
        eeg = eeg.transpose(0, 1)
        eeg = eeg.transpose(0, 1)
        # train_eeg = train_eeg[:, 20:460]
        # x = np.linspace(0, 1, train_eeg.shape[-1])
        # x2 = np.linspace(0, 1, 512)
        # f = interp1d(x, train_eeg)
        # train_eeg = torch.tensor(f(x2))
        # train_eeg = train_eeg.transpose(0, 1)
        # train_eeg = train_eeg.transpose(0, 1)
        _, _, scores = timefreq_model(eeg.to('cuda').to(torch.float32).reshape(1, 128, 512))
        _, pred = torch.topk(scores, 1)
        pred = pred.view(-1).tolist()
        # lastrep, rep, scores = timefreq_model(train_eeg.to('cuda').to(torch.float32).reshape(1, 128, 512))
        # _, train_pred = torch.topk(scores, 1)
        # train_pred = train_pred.view(-1).tolist()
        pres.append(pred)
        # train_pres.append(train_pred)
        print(i)
    pres = [item for sublist in pres for item in sublist]
    pres = np.array(pres)
    for i in range(len(pres)):
        print(propmt_dict[lable_number_dict[str(pres[i])]])
        prompts.append(propmt_dict[lable_number_dict[str(pres[i])]])
    torch.save(prompts, "eval/img/4layers-sbj1/prompts")


main()
