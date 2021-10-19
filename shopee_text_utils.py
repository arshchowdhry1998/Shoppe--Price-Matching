
import numpy as np 
import pandas as pd 
import os
import time
from sentence_transformers import SentenceTransformer, SentencesDataset, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from sentence_transformers import evaluation 
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

class sbert_ensemble:
    def __init__(self,model_paths,weights,ensemble_type='mean_of_embeddings'):
        """
        model_paths supports both - path of sentence transformer model, instance of any model with encode properties
        """
        self.model_paths=model_paths
        self.weights=weights
        self.ensemble_type=ensemble_type
        
    def encode(self,texts):
        embeddings = None
        if self.ensemble_type == 'mean_of_embeddings':
            for i,model_path in enumerate(self.model_paths):
                if type(model_path)==str:
                    model = SentenceTransformer(model_path)
                else:
                    model = model_path
                if i==0:
                    embeddings = model.encode(texts)*self.weights[i]
                else:
                    embeddings += model.encode(texts)*self.weights[i]
                    del model
                    gc.collect()
        if self.ensemble_type == 'concat_of_embeddings':
            embeddings = []
            for i,model_path in enumerate(self.model_paths):
                if type(model_path)==str:
                    model = SentenceTransformer(model_path)
                else:
                    model = model_path
                embeddings.append(model.encode(texts)*self.weights[i])
                del model
                gc.collect()
            embeddings = np.concatenate(embeddings,axis=1)
        return embeddings
    
    def get_neighbours(self,df,KNN=50):
        self.titles = df.title.values
        self.ids = df.posting_id.values
        text_embeddings = self.encode(self.titles)
        neighbors_model = NearestNeighbors(n_neighbors = KNN, metric = 'cosine',n_jobs=-1)
        neighbors_model.fit(text_embeddings)
        text_distances, text_indices = neighbors_model.kneighbors(text_embeddings)
        del neighbors_model,text_embeddings
        gc.collect()
        text_distances = np.abs(text_distances)
        text_neighbours = pd.DataFrame(np.stack([text_indices.reshape(-1),text_distances.reshape(-1)],axis=1),columns=['posting_id2','text_distance'])
        text_neighbours['posting_id1'] = text_neighbours.index//KNN
        text_neighbours = text_neighbours[['posting_id1','posting_id2','text_distance']]
        text_neighbours['posting_id1'] = text_neighbours['posting_id1'].apply(lambda x:self.ids[x])
        text_neighbours['posting_id2'] = text_neighbours['posting_id2'].astype(int).apply(lambda x:self.ids[x])
        del text_distances,text_indices
        gc.collect()
        return text_neighbours
    
    def get_predictions(self,df,KNN=50,threshold=0.25):
        text_neighbours = self.get_neighbours(df,KNN)
        text_neighbours = text_neighbours[text_neighbours.text_distance<threshold].reset_index(drop=True)
        return text_neighbours
    
    def compute_cv(self,df,KNN=50):
        true_pairs = pd.merge(df,df,on='label_group')
        text_neighbours = self.get_neighbours(df,KNN)
        text_neighbours['tp'] = (text_neighbours.posting_id1+text_neighbours.posting_id2).isin(true_pairs.posting_id_x+true_pairs.posting_id_y)
        th_opt,precision_opt,recall_opt,f1_opt = 0,0,0,0
        for threshold in tqdm([0.01*x for x in range(100)]):
            subset = text_neighbours[text_neighbours.text_distance<threshold].reset_index(drop=True)
            precision = subset.groupby('posting_id1').tp.mean()
            recall = subset.groupby('posting_id1').tp.sum()/true_pairs.groupby('posting_id_x').posting_id_y.count()
            f1 = 2*precision*recall/(precision+recall)
            if f1.mean()>f1_opt:
                th_opt,precision_opt,recall_opt,f1_opt = threshold, precision.mean(),recall.mean(),f1.mean()
        print(th_opt,precision_opt,recall_opt,f1_opt)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm
import gc
import re

# Function to get our text title embeddings
def get_text_embeddings(df, max_features = 50500):
    model = TfidfVectorizer(stop_words = 'english', binary = True, max_features = max_features)
    text_embeddings = model.fit_transform(df['title'])
    del model
    gc.collect()
    return text_embeddings

def get_tf_idf_neighbours(test_df,test_ids,max_features=100,KNN=50):
    text_embeddings = get_text_embeddings(test_df,max_features=max_features)
    gc.collect()
    print("Text Embeddings Shape:",text_embeddings.shape)
    neighbors_model = NearestNeighbors(n_neighbors = KNN, metric = 'cosine',n_jobs=-1)
    neighbors_model.fit(text_embeddings)
    text_distances, text_indices = neighbors_model.kneighbors(text_embeddings)
    text_distances = np.abs(text_distances)
    del text_embeddings,neighbors_model
    gc.collect()

    text_neighbours = pd.DataFrame(np.stack([text_indices.reshape(-1),text_distances.reshape(-1)],axis=1),columns=['posting_id2','text_distance'])
    text_neighbours['posting_id1'] = text_neighbours.index//KNN
    text_neighbours = text_neighbours[['posting_id1','posting_id2','text_distance']]
    text_neighbours = text_neighbours[text_neighbours.posting_id1!=text_neighbours.posting_id2].reset_index(drop=True)
    text_neighbours_identity = pd.DataFrame(np.stack([list(range(len(text_indices))),list(range(len(text_indices)))],axis=1),columns=['posting_id1','posting_id2'])
    text_neighbours_identity['text_distance'] = 0
    text_neighbours = pd.concat([text_neighbours_identity,text_neighbours],axis=0).reset_index(drop=True)
    text_neighbours['posting_id1'] = text_neighbours['posting_id1'].apply(lambda x:test_ids[x])
    text_neighbours['posting_id2'] = text_neighbours['posting_id2'].astype(int).apply(lambda x:test_ids[x])
    del text_indices,text_distances,text_neighbours_identity
    gc.collect()
    return text_neighbours

def get_tf_idf_predictions(test_df):
    test_df['title'] = test_df.title.apply(lambda x: x.encode().decode('unicode_escape')).apply(lambda x: re.sub(r'[^a-zA-Z0-9]', ' ', x))
    test_df['title'] = test_df.title.apply(lambda x: re.sub(' +', ' ', x).strip().lower())
    test_ids = test_df.posting_id.values
    print(test_df.shape)

    ### Compute Vocabulary
    Vocab = {}
    for i,row in tqdm(test_df.iterrows()):
        for word in set(row.title.split(' ')):
            if word in Vocab:
                Vocab[word] += 1
            else:
                Vocab[word] = 1

    for i,row in tqdm(test_df.iterrows()):
        words = []
        for word in row.title.split(' '):
            if Vocab[word]>1:
                words.append(word)
        test_df.loc[i,'title'] = " ".join(words)
    test_df['title'] = test_df.title.apply(lambda x: re.sub(' +', ' ', x).strip().lower())

    text_neighbours = get_tf_idf_neighbours(test_df,test_ids,max_features=100000,KNN=min(50,test_df.shape[0]))
    return text_neighbours
