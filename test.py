import gc
import random
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np # linear algebra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
MAX_SEQ =80


train_df = pd.read_csv('/home/pooja/PycharmProjects/kaggle_ventilator/data/ventilator-pressure-prediction/train.csv')
train_df = train_df[train_df.u_out ==0]
#train_df=train_df[train_df.R==5][train_df.C==20]
#arrange by timestamp
train_df = train_df.sort_values(['breath_id','time_step'], ascending=True).reset_index(drop = True)
#preprocessing
train_df.pressure=train_df.pressure/100
train_df.u_in=train_df.u_in/100
group = train_df[['breath_id', 'u_in', 'pressure','time_step']].groupby('breath_id').apply(lambda r: (
            r['u_in'].values,
            r['pressure'].values,
            r['time_step'].values,
            r['R'].values,
            r['C'].values



))


class SAKTDataset(Dataset):
    def __init__(self, group, max_seq=MAX_SEQ):  # HDKIM 100
        super(SAKTDataset, self).__init__()
        self.max_seq = max_seq
        # self.n_skill = n_skill
        self.samples = group
        self.user_ids = [x for x in group.index]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, qa_, ts_,R,C = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq)
        qa = np.zeros(self.max_seq)
        ts = np.zeros(self.max_seq)

        q[-seq_len:] = q_[0:seq_len]
        qa[-seq_len:] = qa_[0:seq_len]
        ts[-seq_len:] = ts_[0:seq_len]

        target_id = q[1:]
        label = qa[1:]

        #         x = np.zeros(self.max_seq-1, dtype=int)
        #         x = q[:-1].copy()
        #         x += (qa[:-1] == 1) * self.n_skill

        return qa[:-1], target_id, label, ts[:-1]
dataset = SAKTDataset(group)
dataloader = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8)


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class SAKTModel(nn.Module):
    def __init__(self, max_seq=MAX_SEQ, embed_dim=1):  # HDKIM 100->MAX_SEQ
        super(SAKTModel, self).__init__()

        self.multi_att = nn.MultiheadAttention(embed_dim=1, num_heads=1, dropout=0.2)

        self.dropout = nn.Dropout(0.2)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids, ts):
        device = x.device
        e = question_ids
        pos_x = ts
        x = x #+ pos_x

        x = x.reshape(x.shape[0], x.shape[1], 1).permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.reshape(e.shape[0], e.shape[1], 1).permute(1, 0, 2)
        att_mask = future_mask(x.size(0)).to(device)
        # print(x.shape,e.shape,att_mask.shape)
        att_output, att_weight = self.multi_att(e, x, x, attn_mask=att_mask)
        #print(att_output.shape)
        att_output = self.layer_normal(att_output + e)

        att_output = att_output.permute(1, 0, 2)  # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = torch.sigmoid(self.pred(x))

        return x.squeeze(-1), att_weight


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAKTModel()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.L1Loss()

model.to(device)
criterion.to(device)


def train_epoch(model, train_iterator, optim, criterion, device="cpu"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    tbar = tqdm(train_iterator)
    for item in tbar:
        x = item[0].to(device).float()
        target_id = item[1].to(device).float()
        label = item[2].to(device).float()
        ts = item[3].to(device).float()

        optim.zero_grad()
        output, atten_weight = model(x, target_id, ts)
        loss = criterion(output, label)*100
        loss.backward()
        optim.step()
        train_loss.append(loss.item())

        #         output = output[:, -1]
        #         label = label[:, -1]

        #         num_corrects += (pred == label).sum().item()
        #         num_total += len(label)

        #         labels.extend(label.view(-1).data.cpu().numpy())
        #         outs.extend(output.view(-1).data.cpu().numpy())

        tbar.set_description('loss - {:.4f}'.format(loss))

    #     acc = num_corrects / num_total
    #     auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss
epochs = 350 #HDKIM 20
for epoch in range(epochs):
    loss= train_epoch(model, dataloader, optimizer, criterion, device)
    print("epoch - {} train_loss - {:.2f} acc ".format(epoch, loss))