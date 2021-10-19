
import sys
sys.path.append('../input/timm-pytorch-image-models/pytorch-image-models-master')

from sklearn.neighbors import NearestNeighbors

import numpy as np 
import pandas as pd 

import gc
import os 
import cv2 
import timm

import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
import torch.nn.functional as F 
from torch import nn 
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parameter import Parameter

import math

from tqdm.notebook import tqdm 
from sklearn.preprocessing import LabelEncoder

from transformers import AdamW
    
class ShopeeDataset(torch.utils.data.Dataset):

    def __init__(self,df, Config, transform = None):
        self.df = df 
        self.root_dir = Config.DATA_DIR
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(self.root_dir,row.image)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         label = row.label_group

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return {
            'image' : image,
            'label' : torch.tensor(0).long()
        }

def get_val_transforms(Config):
    return albumentations.Compose(
        [
            albumentations.Resize(Config.IMG_SIZE,Config.IMG_SIZE,always_apply=True),
            albumentations.Normalize(mean = Config.MEAN, std = Config.STD),
            ToTensorV2(p=1.0),
        ]
    )

class Mish_func(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(F.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1./h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        
        grad_hx = i.sigmoid()

        grad_gx = grad_gh *  grad_hx #grad_hv * grad_vx 
        
        grad_f =  torch.tanh(F.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 


class Mish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        print("Mish initialized")
        pass
    def forward(self, input_tensor):
        return Mish_func.apply(input_tensor)
    
def replace_activations(model, existing_layer, new_layer):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            model._modules[name] = replace_activations(module, existing_layer, new_layer)

        if type(module) == existing_layer:
            layer_old = module
            layer_new = new_layer
            model._modules[name] = layer_new
    return model

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, scale=30, margin=0.5, easy_margin=False, ls_eps=0.0):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.ls_eps = ls_eps  # label smoothing
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

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
        output *= self.scale

        return output, nn.CrossEntropyLoss()(output,label)

class ShopeeModel(nn.Module):

    def __init__(
        self,
        Config):
        
        n_classes = Config.CLASSES
        model_name = Config.MODEL_NAME
        fc_dim = Config.FC_DIM
        margin = Config.MARGIN
        scale = Config.SCALE
        use_fc = Config.use_fc
        pretrained = False


        super(ShopeeModel,self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        if model_name == 'resnext50_32x4d':
            final_in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            self.backbone.global_pool = nn.Identity()

        elif 'efficientnet' in model_name:
            final_in_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
            self.backbone.global_pool = nn.Identity()
        
        elif 'nfnet' in model_name:
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        elif model_name == 'eca_nfnet_l0':
            final_in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
            self.backbone.head.global_pool = nn.Identity()

        if Config.pool == "gem":
            self.pooling = GeM(p_trainable=Config.p_trainable)
        elif Config.pool == "identity":
            self.pooling = torch.nn.Identity()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc

        if use_fc:
            self.dropout = nn.Dropout(p=0.0)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.final = ArcMarginProduct(
            final_in_features,
            n_classes,
            scale = scale,
            margin = margin,
            easy_margin = False,
            ls_eps = 0.0
        )

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, image, label):
        feature = self.extract_feat(image)
        logits = self.final(feature,label)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)
        x = torch.nn.SiLU()(x)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)
        return x
    
def inference_model(model,data_loader,Config):

    model.eval()
    tk = tqdm(data_loader, desc = "Epoch" + " [VALID] ")

    F = []
    with torch.no_grad():
        for t,data in enumerate(tk):
            data = data['image'].to(Config.DEVICE)
            F.append( model.extract_feat(data).detach().cpu().numpy() )

    return np.concatenate(F, axis=0)

def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM,self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1)*p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'
    
def run_inference(Config):

    test_df = Config.TEST_CSV.copy()
    ids = test_df.posting_id.values
    valset = ShopeeDataset(test_df, Config, transform = get_val_transforms(Config))

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size = Config.BATCH_SIZE,
        pin_memory = True,
        num_workers = Config.NUM_WORKERS,
        shuffle = False,
        drop_last = False
    )

    model = ShopeeModel(Config)
    
    if Config.MODEL_PATH!=None:
        model.load_state_dict(torch.load(Config.MODEL_PATH))
    
    model.to(Config.DEVICE)
    
    if Config.mish:
        existing_layer = torch.nn.SiLU
        new_layer = Mish()
        model = replace_activations(model, existing_layer, new_layer) # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()

    embeddings = inference_model(model, valloader, Config)
    
    del model, valloader, test_df, valset
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings,ids

def get_nfnet_neighbours(Config1,Config2=None, KNN=50):
    image_embeddings,ids = run_inference(Config1)
    if Config2:
        image_embeddings2,_ = run_inference(Config2)
        image_embeddings = np.concatenate([image_embeddings,image_embeddings2],axis=1)
        del image_embeddings2
        gc.collect()
    print("Embeddings Shape:",image_embeddings.shape)
    neighbors_model = NearestNeighbors(n_neighbors = KNN, metric = 'cosine',n_jobs=-1)
    neighbors_model.fit(image_embeddings)
    image_distances, image_indices = neighbors_model.kneighbors(image_embeddings)
    image_distances = np.abs(image_distances)
    del image_embeddings,neighbors_model
    gc.collect()
    image_neighbours = pd.DataFrame(np.stack([image_indices.reshape(-1),image_distances.reshape(-1)],axis=1),columns=['posting_id2','nfnet_distance'])
    image_neighbours['posting_id1'] = image_neighbours.index//KNN
    image_neighbours = image_neighbours[['posting_id1','posting_id2','nfnet_distance']]
    image_neighbours['posting_id1'] = image_neighbours['posting_id1'].apply(lambda x:ids[x])
    image_neighbours['posting_id2'] = image_neighbours['posting_id2'].astype(int).apply(lambda x:ids[x])
    del image_indices,image_distances
    gc.collect()
    return image_neighbours
