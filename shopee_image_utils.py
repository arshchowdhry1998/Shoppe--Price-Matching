import re
import os
import gc
import numpy as np
import pandas as pd
import random
import math
import tensorflow as tf
import efficientnet.tfkeras as efn
from sklearn import metrics
import dask.dataframe as dd
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tqdm.notebook import tqdm
from kaggle_datasets import KaggleDatasets
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from IPython.display import display
from numba import cuda 

strategy = tf.distribute.get_strategy()
AUTO = tf.data.experimental.AUTOTUNE
VERBOSE = 1
# Function to decode our images
def decode_image(image_data,IMAGE_SIZE):
    image = tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image

# Function to read our test image and return image
def read_image(image,IMAGE_SIZE):
    image = tf.io.read_file(image)
    image = decode_image(image,IMAGE_SIZE)
    return image

# Function to get our dataset that read images
def get_dataset(df,IMAGE_SIZE,BATCH_SIZE=32):
    dataset = tf.data.Dataset.from_tensor_slices((df.posting_id.values,df.image_path.values))
    dataset = dataset.map(lambda posting_id,image_path: (posting_id,read_image(image_path,IMAGE_SIZE)), num_parallel_calls = AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


# Arcmarginproduct class keras layer
class ArcMarginProduct(tf.keras.layers.Layer):
    '''
    Implements large margin arc distance.

    Reference:
        https://arxiv.org/pdf/1801.07698.pdf
        https://github.com/lyakaap/Landmark2019-1st-and-3rd-Place-Solution/
            blob/master/src/modeling/metric_learning.py
    '''
    def __init__(self, n_classes, s=30, m=0.50, easy_margin=False,
                 ls_eps=0.0, **kwargs):

        super(ArcMarginProduct, self).__init__(**kwargs)

        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.ls_eps = ls_eps
        self.easy_margin = easy_margin
        self.cos_m = tf.math.cos(m)
        self.sin_m = tf.math.sin(m)
        self.th = tf.math.cos(math.pi - m)
        self.mm = tf.math.sin(math.pi - m) * m

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'ls_eps': self.ls_eps,
            'easy_margin': self.easy_margin,
        })
        return config

    def build(self, input_shape):
        super(ArcMarginProduct, self).build(input_shape[0])

        self.W = self.add_weight(
            name='W',
            shape=(int(input_shape[0][-1]), self.n_classes),
            initializer='glorot_uniform',
            dtype='float32',
            trainable=True,
            regularizer=None)

    def call(self, inputs):
        X, y = inputs
        y = tf.cast(y, dtype=tf.int32)
        cosine = tf.matmul(
            tf.math.l2_normalize(X, axis=1),
            tf.math.l2_normalize(self.W, axis=0)
        )
        sine = tf.math.sqrt(1.0 - tf.math.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = tf.where(cosine > 0, phi, cosine)
        else:
            phi = tf.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = tf.cast(
            tf.one_hot(y, depth=self.n_classes),
            dtype=cosine.dtype
        )
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.n_classes

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

def freeze_BN(model):
    # Unfreeze layers while leaving BatchNorm layers frozen
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
        else:
            layer.trainable = False

# Function to create our EfficientNetB3 model
def get_model(EFF_NET,IMAGE_SIZE,N_CLASSES=11014):

    with strategy.scope():

        margin = ArcMarginProduct(
            n_classes = N_CLASSES, 
            s = 30, 
            m = 0.5, 
            name='head/arc_margin', 
            dtype='float32'
            )

        inp = tf.keras.layers.Input(shape = (*IMAGE_SIZE, 3), name = 'inp1')
        label = tf.keras.layers.Input(shape = (), name = 'inp2')
        x = EFNS[EFF_NET](weights = None, include_top = False)(inp)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = margin([x, label])
        
        output = tf.keras.layers.Softmax(dtype='float32')(x)

        model = tf.keras.models.Model(inputs = [inp, label], outputs = [output])
        
        return model
    
def get_image_embeddings(model_weight,df,IMAGE_SIZE=[512,512],BATCH_SIZE = 32):
    EFF_NET,fold,epoch = list(map(int,model_weight.split('/')[-1].replace('.h5','').replace('EF','').replace('fold','').replace('epoch','').split('_')))
    model = get_model(EFF_NET,IMAGE_SIZE)
    model.load_weights(model_weight)
    embed_model = tf.keras.models.Model(inputs = model.input[0], outputs = model.layers[-4].output)
    test_dataset = get_dataset(df,IMAGE_SIZE,BATCH_SIZE=BATCH_SIZE)
    ids = df.posting_id.values
    test_dataset = test_dataset.map(lambda posting_id, image: image)
    image_embeddings = embed_model.predict(test_dataset,verbose=VERBOSE)
    del model,test_dataset
    gc.collect()
    return image_embeddings,ids

def ensemble_image_embeddings(model_weights,df,IMAGE_SIZES=None,BATCH_SIZE = 32,ensemble_type='concat'):
    if ensemble_type=='concat':
        all_image_embeddings = []
        for i,model_weight in enumerate(model_weights):
                IMAGE_SIZE = IMAGE_SIZES[model_weight]
                image_embeddings,ids = get_image_embeddings(model_weight,df,IMAGE_SIZE=IMAGE_SIZE,BATCH_SIZE = BATCH_SIZE)
                all_image_embeddings.append(image_embeddings * model_weights[model_weight])
        all_image_embeddings = np.concatenate(all_image_embeddings,axis=1)
    elif ensemble_type=='mean':
        for i,model_weight in enumerate(model_weights):
            IMAGE_SIZE = IMAGE_SIZES[model_weight]
            image_embeddings,ids = get_image_embeddings(model_weight,df,IMAGE_SIZE=IMAGE_SIZE,BATCH_SIZE = BATCH_SIZE)
            if i==0:
                all_image_embeddings = image_embeddings * model_weights[model_weight]
            else:
                all_image_embeddings += image_embeddings * model_weights[model_weight]
    return all_image_embeddings,ids
        
def get_image_neighbours(model_weights,df,IMAGE_SIZES=None,BATCH_SIZE = 32,KNN=50,ensemble_type='concat'):
    image_embeddings,ids = ensemble_image_embeddings(model_weights,df,IMAGE_SIZES=IMAGE_SIZES,BATCH_SIZE = BATCH_SIZE,ensemble_type=ensemble_type)
    print("Embeddings Shape:",image_embeddings.shape)
    neighbors_model = NearestNeighbors(n_neighbors = KNN, metric = 'cosine',n_jobs=-1)
    neighbors_model.fit(image_embeddings)
    image_distances, image_indices = neighbors_model.kneighbors(image_embeddings)
    image_distances = np.abs(image_distances)
    del image_embeddings
    gc.collect()
    image_neighbours = pd.DataFrame(np.stack([image_indices.reshape(-1),image_distances.reshape(-1)],axis=1),columns=['posting_id2','image_distance'])
    image_neighbours['posting_id1'] = image_neighbours.index//KNN
    image_neighbours = image_neighbours[['posting_id1','posting_id2','image_distance']]
    image_neighbours['posting_id1'] = image_neighbours['posting_id1'].apply(lambda x:ids[x])
    image_neighbours['posting_id2'] = image_neighbours['posting_id2'].astype(int).apply(lambda x:ids[x])
    del image_indices,image_distances
    gc.collect()
    return image_neighbours
