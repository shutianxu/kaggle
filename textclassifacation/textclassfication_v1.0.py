"""
Created on Tue Apr 10 14:22:48 2018
@author: 1707500
"""
'''
V1.0 vocab + glove/fasttest + rnn 训练
问题：原始训练数据已经编码处理，以训练好的glove词向量无法匹配，暂定用TF-IDF尝试

'''


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
import collections
import gluonbook as gb
from mxnet import autograd, gluon, init, metric, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
from time import time
import tarfile



column = "word_seg"
train = pd.read_csv('D:/data/new_data/train_set.csv')
test = pd.read_csv('D:/data/new_data/test_set.csv')
test_id = test["id"].copy()

train_tokenized = []
for i in train['word_seg']:
    train_tokenized.append(i.split(' '))
    
test_tokenized = []
for j in test['word_seg']:
    train_tokenized.append(j.split(' '))

token_counter = collections.Counter()
def count_token(train_tokenized):
    for sample in train_tokenized:
        for token in sample:
            if token not in token_counter:
                token_counter[token] = 1
            else:
                token_counter[token] += 1



count_token(train_tokenized)
vocab = text.vocab.Vocabulary(token_counter, unknown_token='<unk>',reserved_tokens=None)



def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in vocab.token_to_idx:
                feature.append(vocab.token_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            # 添加 PAD 符号使每个序列等长（长度为 maxlen）。
            while len(padded_feature) < maxlen:
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features

ctx = gb.try_gpu()
train_features = encode_samples(train_tokenized, vocab)
test_features = encode_samples(test_tokenized, vocab)
train_features = nd.array(pad_samples(train_features, 500, 0), ctx=ctx)
test_features = nd.array(pad_samples(test_features, 500, 0), ctx=ctx)
train_labels = nd.array(train['class'], ctx=ctx)
# =============================================================================
# test_labels = nd.array(test['class'], ctx=ctx)
# =============================================================================



# =============================================================================
# glove_embedding = text.embedding.create(
#     'glove', pretrained_file_name='glove.6B.50d.txt', vocabulary=vocab)
# =============================================================================

glove_embedding = text.embedding.create(
        'fasttext', pretrained_file_name='wiki.simple.vec',vocabulary=vocab)



class SentimentNet(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers,
                 bidirectional, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        with self.name_scope():
            self.embedding = nn.Embedding(len(vocab), embed_size)
            self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    input_size=embed_size)
            self.decoder = nn.Dense(num_outputs, flatten=False)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.T)
        states = self.encoder(embeddings)
        # 连结初始时间步和最终时间步的隐藏状态。
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
    
    
num_outputs = 2
lr = 0.8
num_epochs = 5
batch_size = 64
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
ctx = gb.try_all_gpus()

net = SentimentNet(vocab, embed_size, num_hiddens, num_layers, bidirectional)
net.initialize(init.Xavier(), ctx=ctx)
# 设置 embedding 层的 weight 为预训练的词向量。
# =============================================================================
# net.embedding.weight.set_data(glove_embedding.idx_to_vec)
# =============================================================================
# 训练中不更新词向量（net.embedding 中的模型参数）。
net.embedding.collect_params().setattr('grad_req', 'null')
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()



train_set = gdata.ArrayDataset(train_features, train_labels)
test_set = gdata.ArrayDataset(test_features, test_labels)
train_loader = gdata.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True)
test_loader = gdata.DataLoader(test_set, batch_size=batch_size, shuffle=False)

gb.train(train_loader, test_loader, net, loss, trainer, ctx, num_epochs)
