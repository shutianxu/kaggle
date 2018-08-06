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
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm


column = "word_seg"
train = pd.read_csv('D:/data/new_data/train_set.csv')
test = pd.read_csv('D:/data/new_data/test_set.csv')
test_id = test["id"].copy()


vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])

column="word_seg"
vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train[column])
test_term_doc = vec.transform(test[column])



y=(train["class"]-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc,y)
preds = lin_clf.predict(test_term_doc)
