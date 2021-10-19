
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import clip
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import re
from clip.simple_tokenizer import SimpleTokenizer
import faiss
import torch
import torch.nn as nn

_tokenizer = SimpleTokenizer()

# Copied from https://github.com/openai/CLIP/blob/beba48f35392a73c6c47ae67ddffced81ad1916d/clip/clip.py#L164
# but with relaxed exception
def tokenize(texts, context_length: int = 77) -> torch.LongTensor:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        n = min(len(tokens), context_length)
        result[i, :n] = torch.tensor(tokens)[:n]
        if len(tokens) > context_length:
            result[i, -1] = tokens[-1]

    return result

# Remove EMOJI
RE_EMOJI = re.compile(r"\\x[A-Za-z0-9./]+", flags=re.UNICODE)

def strip_emoji(text):
    return RE_EMOJI.sub(r'', text)

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.clip, self.preprocess = clip.load("../input/openai-clip/ViT-B-32.pt", device='cpu', jit=False)
        
        self.embed_dim = self.clip.text_projection.shape[1]
    
        self.classification_head = nn.Linear(self.embed_dim, n_classes)
    
    def forward(self, images, texts, return_classification=False):
        images_features = self.clip.encode_image(images)
        texts_features = self.clip.encode_text(texts)

        # Average images and text features, because CLIP was trained to align them
        features = l2_normalize(images_features + texts_features)

        if return_classification:
            classification_output = self.classification_head(features)
            
            return features, classification_output
        else:
            return features
        
def l2_normalize(features):
    return features / features.norm(2, dim=1, keepdim=True)

class MyDataset(Dataset):
    def __init__(self, df, images_path, preprocess):
        super().__init__()
        self.df = df
        self.images_path = images_path
        self.preprocess = preprocess
        self.has_target = ('label_group' in df)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        image = self.preprocess(Image.open(self.images_path / row['image']))
        text = tokenize([strip_emoji(row['title'])])[0]
        
        if self.has_target:
            return image, text, row['label_group']
        else:
            return image, text, 0

class ClipModel:
    def __init__(self,model_path,df=None,KNN=50,n_classes=8811,
                 images_path='../input/shopee-product-matching/train_images/'):
        self.model = CLIPClassifier(n_classes).to(device)
        self.ids = df.posting_id.values
        self.df = df.set_index('posting_id')
        self.images_path = Path(images_path)
        self.model_path = model_path
        self.KNN = KNN

    def find_similarities_and_indexes(self):
        # Create pytorch Dataset/DataLoader
        ds = MyDataset(self.df, self.images_path, self.model.preprocess)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

        # Allocate memory for features
        features = np.empty((len(self.df), self.model.embed_dim), dtype=np.float32)

        # Begin predict
        i = 0
        for images, texts, _ in tqdm(dl):
            n = len(images)
            with torch.no_grad():
                # Generate image and text features
                batch_features = self.model(images.to(device), texts.to(device), return_classification=False)

            # Average images and text features, because CLIP was trained to align them
            features[i:i+n] = batch_features.cpu()

            i += n

        # Create index
        index = faiss.IndexFlatIP(features.shape[1])
        index.add(features)

        # Search index
        return index.search(features, self.KNN)

    def get_neighbours(self):
        self.model.load_state_dict(torch.load(self.model_path))
        clip_distances, clip_indices = self.find_similarities_and_indexes()
        clip_distances = 1-clip_distances
        clip_neighbours = pd.DataFrame(np.stack([clip_indices.reshape(-1),clip_distances.reshape(-1)],axis=1),columns=['posting_id2','clip_distance'])
        clip_neighbours['posting_id1'] = clip_neighbours.index//self.KNN
        clip_neighbours = clip_neighbours[['posting_id1','posting_id2','clip_distance']]
        clip_neighbours['posting_id1'] = clip_neighbours['posting_id1'].apply(lambda x:self.ids[x])
        clip_neighbours['posting_id2'] = clip_neighbours['posting_id2'].astype(int).apply(lambda x:self.ids[x])
        return clip_neighbours
