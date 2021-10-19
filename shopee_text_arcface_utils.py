
import math
from tqdm.auto import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Visuals and CV2
import cv2
import gc

# albumentations for augs
import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import KFold, train_test_split
import gc
#torch
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam, lr_scheduler

import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

NUM_WORKERS = 4
TRAIN_BATCH_SIZE = 32
EPOCHS = 25
SEED = 2020
LR = 5e-5

device = torch.device('cuda')

################################################# MODEL ####################################################################

transformer_model = '/kaggle/input/sentence-transformer-models/paraphrase-xlm-r-multilingual-v1/0_Transformer'
TOKENIZER = transformers.AutoTokenizer.from_pretrained(transformer_model)

################################################ Metric Loss and its params #######################################################
loss_module = 'arcface'#'softmax'
s = 30.0
m = 0.5 
ls_eps = 0.0
easy_margin = False

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.ls_eps = ls_eps  # label smoothing
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
    
class ShopeeNet(nn.Module):

    def __init__(self,
                 n_classes,
                 model_name='bert-base-uncased',
                 pooling='mean_pooling',
                 use_fc=False,
                 fc_dim=512,
                 dropout=0.0,
                 loss_module='softmax',
                 s=30.0,
                 margin=0.50,
                 ls_eps=0.0,
                 theta_zero=0.785):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param loss_module: One of ('arcface', 'cosface', 'softmax')
        """
        super(ShopeeNet, self).__init__()

        self.transformer = transformers.AutoModel.from_pretrained(transformer_model)
        final_in_features = self.transformer.config.hidden_size
        
        self.pooling = pooling
        self.use_fc = use_fc
    
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self.relu = nn.ReLU()
            self._init_params()
            final_in_features = fc_dim

        self.loss_module = loss_module
        if loss_module == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        
    def get_embeddings(self, input_ids,attention_mask):
        feature = self.extract_feat(input_ids,attention_mask)
        return F.normalize(feature)

    def forward(self, input_ids,attention_mask, label):
        feature = self.extract_feat(input_ids,attention_mask)
        if self.loss_module == 'arcface':
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, input_ids,attention_mask):
        x = self.transformer(input_ids=input_ids,attention_mask=attention_mask)
        
        features = x[0]
        features = features[:,0,:]

        if self.use_fc:
            features = self.dropout(features)
            features = self.fc(features)
            features = self.bn(features)
            features = self.relu(features)

        return features

class ShopeeDataset(Dataset):
    def __init__(self, csv):
        self.csv = csv.reset_index()

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]
        
        text = row.title
        
        text = TOKENIZER(text, padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = text['input_ids'][0]
        attention_mask = text['attention_mask'][0]  
        
        return input_ids, attention_mask

class arface_model:
    def __init__(self,model_params,model_weights):
        # Defining Device
        device = torch.device("cuda")
        self.BATCH_SIZE = 16

        # Defining Model for specific fold
        self.model = ShopeeNet(**model_params)
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_weights))
    
    def encode(self,texts):
        df = pd.DataFrame(texts,columns=['title'])
        embeds = []

        text_dataset = ShopeeDataset(df)
        text_loader = torch.utils.data.DataLoader(
            text_dataset,
            batch_size=self.BATCH_SIZE,
            pin_memory=True,
            drop_last=False,
            num_workers=NUM_WORKERS
        )


        with torch.no_grad():
            for input_ids, attention_mask in tqdm(text_loader): 
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                feat = self.model.get_embeddings(input_ids, attention_mask)
                text_embeddings = feat.detach().cpu().numpy()
                embeds.append(text_embeddings)


        text_embeddings = np.concatenate(embeds)
        del embeds,text_loader,text_dataset
        gc.collect()
        return text_embeddings
