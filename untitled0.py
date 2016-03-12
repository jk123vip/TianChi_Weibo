# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:59:05 2015

@author: Administrator

测试stopwords
"""

import numpy as np
import jieba
import string

# 解决中文编码问题
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 停用词列表
stopwords = []
f = open('stopwords.txt', 'r')
for line in f.readlines():
    stopwords.append(line.decode('gbk').strip('\n'))

data = []
f = open('weibo_train_data.txt')
for line in f:
    data.append(line)

texts = [record.split('\t')[6] for record in data]
texts_filtered = [document.translate(None, string.punctuation+' '+'\n'+string.digits+string.ascii_letters) \
                for document in texts]
texts_filtered1 = [jieba.lcut(document) for document in texts_filtered]
texts_words = [[word for word in document if word not in stopwords] for document in texts_filtered1]
#texts_filtered2 = [[word for word in document if word not in stopwords] for document in texts_filtered]
#texts = [jieba.lcut(document) for document in a]
