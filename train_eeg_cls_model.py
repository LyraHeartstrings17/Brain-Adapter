import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from loss import CE
from model.BBBModels import ConvFreqEncoder, ConvLSTMCell, ConvLSTM, FreqEncoder, TimeFreqEncoder, TimeEncoder, \
    OnlyTimeEncoder, OnlyFreqEncoder
from BBB_Align import AlignDataset, Splitter


def get_rep_with_label(model, dataloader):
    reps = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = [x.to('cuda').to(torch.float32) for x in batch]
            seq, label, ip_emb, neg_ip_emb = batch
            seq = seq.to('cuda')
            labels += label.cpu().numpy().tolist()
            rep = model(seq)
            reps += rep.cpu().numpy().tolist()
    return reps, labels


def compute_metrics_freq(batch, model):
    #if len(batch) == 2:
    seqs, label = batch
    scores = model(seqs)  # 记得改回来lastrep, rep,
    top3acc = top_k_accuracy_score(y_true=label, y_pred=scores, k=3)
    top5acc = top_k_accuracy_score(y_true=label, y_pred=scores, k=5)
    #else:
    #    seqs1, seqs2, label = batch
    #    lastrep, rep, scores = self.model((seqs1, seqs2))
    _, pred = torch.topk(scores, 1)
    #print(np.shape(scores))
    test_cr = torch.nn.CrossEntropyLoss()
    test_loss = test_cr(scores, label.view(-1).long())
    pred = pred.view(-1).tolist()
    return pred, label.tolist(), test_loss, top3acc, top5acc


def _confusion_mat(label, pred):
    mat = np.zeros((40, 40))
    for _label, _pred in zip(label, pred):
        _pred = int(_pred)
        _label = int(_label)
        mat[_label, _pred] += 1
    return mat


def top_k_accuracy_score(y_true, y_pred, k=5):
    top_k_preds = torch.topk(y_pred, k, dim=1)[1]
    correctness = top_k_preds.eq(y_true.view(-1, 1).expand_as(top_k_preds))
    top_k_accuracy = correctness.sum().float() / y_true.size(0)
    return top_k_accuracy.item()  # 返回一个Python标量


def fit_lr(features, y):
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            random_state=3407,
            max_iter=1000000,
            multi_class='ovr'
        )
    )
    pipe.fit(features, y)
    return pipe


def print_process(self, *x):
    print(*x)


def finetune_timefreq():
    dataset = AlignDataset(eeg_signals_path='',
                           subject=5)
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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    time_model = TimeEncoder()
    metafile = torch.load(
        "./pre_train_model/checkpoint.pth",
        map_location='cpu')
    time_model.load_checkpoint(metafile['model'])
    time_model.to('cuda')
    print("freq_train")

    time_model.eval()
    time_model.to(torch.device("cuda"))

    time_model.train()
    time_model.to(torch.device("cpu"))
    freq_model = torch.load("./pre_train_model/conv_lstm_4layers__subject0_epoch_1000.pth", map_location='cpu')
    freq_model.to('cuda')
    timefreq_model = OnlyFreqEncoder(freq_model)
    timefreq_model = timefreq_model.to('cuda')

    optimizer = torch.optim.AdamW(timefreq_model.parameters(), lr=0.001)
    cr_freq = CE(timefreq_model)
    eval_acc = 0

    for epoch in range(50):
        print('Epoch:' + str(epoch + 1))
        timefreq_model.train()
        tqdm_dataloader = tqdm(train_loader)
        test_tqdm_dataloader = tqdm(test_loader)
        loss_sum = 0

        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to('cuda').to(torch.float32) for x in batch]
            optimizer.zero_grad()
            loss = cr_freq.computefreq(batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        trloss = loss_sum / (idx + 1)

        metrics = {'acc': 0, 'f1': 0}
        pred = []
        label = []
        test_loss = 0
        acc3 = 0
        acc5 = 0

        for idxte, batch in enumerate(test_tqdm_dataloader):
            timefreq_model.eval()
            batch = [x.to('cuda').to(torch.float32) for x in batch]
            ret = compute_metrics_freq(batch, timefreq_model)
            if len(ret) == 2:
                pred_b, label_b = ret
                pred += pred_b
                label += label_b
            else:
                pred_b, label_b, test_loss_b, acc3_b, acc5_b = ret
                pred += pred_b
                label += label_b
                acc3 += acc3_b
                acc5 += acc5_b
                test_loss += test_loss_b.cpu().item()
        confusion_mat = _confusion_mat(label, pred)
        print_process(confusion_mat)

        metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
        metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
        metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
        metrics['test_loss'] = test_loss / (idxte + 1)

        te_top3acc = acc3 / (idxte + 1)
        te_top5acc = acc5 / (idxte + 1)

        print('timefreq_finetune epoch{0}, trloss{1}, teloss{2},teacc{3},te_top3acc{4},te_top5acc{5}'.format(
            epoch + 1, trloss, metrics['test_loss'], metrics['acc'], te_top3acc, te_top5acc))

        # if (epoch + 1) % 5 == 0:
        #     torch.save(timefreq_model,
        #                './pre_train_model/time_freq_cls_model' + str(epoch + 1) + '.pth')

        if metrics['acc'] > eval_acc:
            eval_acc = metrics['acc']
            torch.save(timefreq_model, './pre_train_model/cls/only_F_cls_model.pth')
        print(eval_acc)


def eval_timefreq():
    dataset = AlignDataset(eeg_signals_path='',
                           subject=0)
    test_dataset = Splitter(dataset,
                            split_path='',
                            split_num=0, split_name="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    timefreq_model = torch.load('./pre_train_model/cls/only_F_cls_model.pth')
    print(len(test_dataset))
    print('eval cls model')
    timefreq_model = timefreq_model.to('cuda')
    metrics = {'acc': 0, 'f1': 0}
    pred = []
    label = []
    test_loss = 0
    acc3 = 0
    acc5 = 0
    test_tqdm_dataloader = tqdm(test_loader)
    for idxte, batch in enumerate(test_tqdm_dataloader):
        timefreq_model.eval()
        batch = [x.to('cuda').to(torch.float32) for x in batch]
        ret = compute_metrics_freq(batch, timefreq_model)
        if len(ret) == 2:
            pred_b, label_b = ret
            pred += pred_b
            label += label_b
        else:
            pred_b, label_b, test_loss_b, acc3_b, acc5_b = ret
            pred += pred_b
            label += label_b
            acc3 += acc3_b
            acc5 += acc5_b
            test_loss += test_loss_b.cpu().item()
    confusion_mat = _confusion_mat(label, pred)
    print_process(confusion_mat)

    metrics['f1'] = f1_score(y_true=label, y_pred=pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true=label, y_pred=pred, average='micro')
    metrics['acc'] = accuracy_score(y_true=label, y_pred=pred)
    metrics['test_loss'] = test_loss / (idxte + 1)

    te_top3acc = acc3 / (idxte + 1)
    te_top5acc = acc5 / (idxte + 1)

    print('best_timefreq_cls teloss{0},teacc{1},te_top3acc{2},te_top5acc{3}, f1score{4}'.format(
        metrics['test_loss'], metrics['acc'], te_top3acc, te_top5acc, metrics['f1']))


if __name__ == '__main__':
    eval_timefreq()
