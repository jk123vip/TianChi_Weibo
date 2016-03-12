# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 17:27:32 2015

@author: Administrator
"""

import numpy as np
import pandas as pd
import jieba
import string
import re
import difflib

# 解决中文编码问题
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 读入训练数据
# 这种方法会少读入4000个数据，所以不采用
#df = pd.read_csv('weibo_train_data.txt', sep='\t', names=['uid', 'mid', 'time', 'f', 'c', 'l', 'co'])
#df[['f', 'c', 'l']] = df[['f', 'c', 'l']].astype(float)
#df.time = pd.to_datetime(df.time)
#df['fenci'] = pd.Series(LoadFenCi())
# 读入训练数据
def LoadTrainData():
    data = []
    f = open('weibo_train_data_new.txt')
    for line in f:
        data.append(line)
    f.close()
    data = pd.Series(data).map(lambda x: x.split('\t'))
    s0 = pd.Series(data.map(lambda x: x[0]), name='uid')
    s1 = pd.Series(data.map(lambda x: x[1]), name='mid')
    s2 = pd.Series(data.map(lambda x: x[2]), name='time')
    s3 = pd.Series(data.map(lambda x: float(x[3])), name='f')
    s4 = pd.Series(data.map(lambda x: float(x[4])), name='c')
    s5 = pd.Series(data.map(lambda x: float(x[5])), name='l')
    s6 = pd.Series(data.map(lambda x: x[6]), name='co')
    df = pd.concat([s0, s1, s2, s3, s4, s5, s6], axis=1)
    df.time = pd.to_datetime(df.time)
    return df

# 读入待预测数据
#df_pre = pd.read_csv('weibo_predict_data.txt', sep='\t', names=['uid', 'mid', 'time', 'co'])
#df_pre.time = pd.to_datetime(df_pre.time)
def LoadPredictData():
    data = []
    f = open('weibo_predict_data_new.txt')
    for line in f:
        data.append(line)
    f.close()
    data = pd.Series(data).map(lambda x: x.split('\t'))
    s0 = pd.Series(data.map(lambda x: x[0]), name='uid')
    s1 = pd.Series(data.map(lambda x: x[1]), name='mid')
    s2 = pd.Series(data.map(lambda x: x[2]), name='time')
    s3 = pd.Series(data.map(lambda x: x[3]), name='co')
    df = pd.concat([s0, s1, s2, s3], axis=1)
    return df

# 法一：010
def Predict1(df_test):
    df_test['fp'] = 0
    df_test['cp'] = 0
    df_test['lp'] = 0

# 法二：中位数
def Predict2(train_df, test_df):
    test_df['fp'] = 0
    test_df['cp'] = 0
    test_df['lp'] = 0
    train_mid = train_df.groupby('uid', as_index=False).median()
    f_dic_train = train_mid.set_index('uid')['f'].to_dict()
    f_dic = test_df.set_index('uid')['fp'].to_dict()
    f_dic.update(f_dic_train)
    c_dic_train = train_mid.set_index('uid')['c'].to_dict()
    c_dic = test_df.set_index('uid')['cp'].to_dict()
    c_dic.update(c_dic_train)
    l_dic_train = train_mid.set_index('uid')['l'].to_dict()
    l_dic = test_df.set_index('uid')['lp'].to_dict()
    l_dic.update(l_dic_train)
    test_df['fp'] = test_df['uid'].map(f_dic).astype(int)
    test_df['cp'] = test_df['uid'].map(c_dic).astype(int)
    test_df['lp'] = test_df['uid'].map(l_dic).astype(int)


# 计算整体准确率
def Precision(df):
    df['prei'] = 1-0.5*abs(df.fp-df.f)/(df.f+5)-0.25*abs(df.cp-df.c)/(df.c+3)-0.25*abs(df.lp-df.l)/(df.l+3)
    df['sgn'] = 0
    df.loc[df.prei-0.8>0, 'sgn'] = 1
    df['coui'] = df.f + df.c + df.l
    df.loc[df.coui>100, 'coui'] = 100
    p = ((df['coui']+1)*df['sgn']).sum() / (df['coui']+1).sum()
    return p

# 对新数据用jieba分词（已经分好，保存在"fenci_new.txt"文件中）
def FenCi(df):
    stopwords = []    # 先读入停用词列表
    f = open('stopwords.txt', 'r')
    for line in f.readlines():
        stopwords.append(line.decode('gbk').strip('\n'))
    f.close()

    documents = [document.translate(None, string.punctuation+' '+'\n'+string.digits+string.ascii_letters) \
                for document in df.co.values]
    texts = [jieba.lcut(document) for document in documents]
    texts = [[word for word in document if word not in stopwords] for document in texts]

    f = open('fenci_new.txt', 'w+')
    for line in texts:
        k = ','.join(line)
        f.write(k+'\n')
    f.close()

# 读入分词结果
def LoadFenCi():
    texts = []
    f = open('fenci_new.txt', 'r')
    for line in f:
        texts.append(line.strip('\n').split(','))
    f.close()
    return texts


# 法三测试，difflib求距离，先用一个uid试试
def TestPredict3(uid_df_train, uid_df_test):
    uid_df_train.index = range(len(uid_df_train))
    uid_df_test.index = range(len(uid_df_test))
    fp = []
    cp = []
    lp = []
    for j in xrange(len(uid_df_test)):
        a = [difflib.SequenceMatcher(None, uid_df_test.fenci.values[j], \
            uid_df_train.fenci.values[i]).ratio() for i in range(len(uid_df_train))]
        b = np.argsort(a)[-3:]
        c = uid_df_train.ix[b]
        fp.append(c.mean().f)
        cp.append(c.mean().c)
        lp.append(c.mean().l)
    uid_df_test.fp = fp
    uid_df_test.cp = cp
    uid_df_test.lp = lp
    print 'TestPredict3: ', Precision(uid_df_test)

# 法三：找K近邻，取均值（用difflib）
def Predict3(df_train, df_test):
    Predict2(df_train, df_test)    # 先Predict2，在其结果上再做Predict3
    df_train_uni = df_train.groupby('uid').mean().sort('f', ascending=False)
#    df_test_uni = df_test.groupby('uid').mean().sort('f', ascending=False)
    uid_train_list = df_train_uni.index.values    # 训练样本所有uid
    uid_test_list = df_test_uni.index.values    # 测试样本所有uid
    f_dic_test = df_test.set_index('mid')['fp'].to_dict()
    c_dic_test = df_test.set_index('mid')['cp'].to_dict()
    l_dic_test = df_test.set_index('mid')['lp'].to_dict()
    num = 0    # 统计处理的样本个数
    for u in xrange(100):#xrange(len(uid_test_list)):    # 40000左右
        uid = uid_train_list[u]
        uid_df_train = df_train[df_train.uid == uid]    # 此uid对应训练样本
        uid_df_test = df_test[df_test.uid == uid]    # 此uid对应测试样本
        if len(uid_df_test)!=0:
            uid_df_train.index = range(len(uid_df_train))    # 重置index
            uid_df_test.index = range(len(uid_df_test))
            for j in xrange(len(uid_df_test)):
                a = [difflib.SequenceMatcher(None, uid_df_test.fenci.values[j], uid_df_train.fenci.values[i]).ratio() for i in range(len(uid_df_train))]
                b = np.argsort(a)[-5:]    # 取最相似的前n个，然后去均值
                c = uid_df_train.ix[b]
                fp = c.mean().f.astype(int)    # 如果fp<20，就把fp赋成0吧
                cp = c.mean().c.astype(int)
                lp = c.mean().l.astype(int)
                if fp < 10:
                    f_dic_test[uid_df_test.mid.values[j]] = 0
                elif fp > 200:
                    f_dic_test[uid_df_test.mid.values[j]] = fp
                if cp < 10:
                    c_dic_test[uid_df_test.mid.values[j]] = 0
                elif cp > 200:
                    c_dic_test[uid_df_test.mid.values[j]] = cp
                if lp < 10:
                    l_dic_test[uid_df_test.mid.values[j]] = 0
                elif lp > 200:
                    l_dic_test[uid_df_test.mid.values[j]] = lp
#                f_dic_test[uid_df_test.mid.values[j]] = fp
#                c_dic_test[uid_df_test.mid.values[j]] = cp
#                l_dic_test[uid_df_test.mid.values[j]] = lp
#                if fp > 500: f_dic_test[uid_df_test.mid.values[j]] = fp
#                if cp > 10: c_dic_test[uid_df_test.mid.values[j]] = cp
#                if lp > 10: l_dic_test[uid_df_test.mid.values[j]] = lp
                num += 1
                # 输出已处理样本个数，正在处理第几个用户，正在处理该用户第几条测试微博，预测的fcl值
                print num, '\t', u+1, '\t', j, '\t', fp, '\t', cp, '\t', lp
    #        uid_df_test.fp = fp
    #        uid_df_test.cp = cp
    #        uid_df_test.lp = lp
    #        for mid in uid_df_test.mid.values:
    #            df_test[df_test.mid == mid].fp = uid_df_test[uid_df_test.mid == mid].fp
    df_test.fp = df_test.mid.map(f_dic_test)
    df_test.cp = df_test.mid.map(c_dic_test)
    df_test.lp = df_test.mid.map(l_dic_test)
    #    print Precision(uid_df_test)

# 法四测试，LSI语义最相关求均值，先对一个用户试试
from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
def TestPredict4(uid_df_train, uid_df_test):
    uid_df_train.index = xrange(len(uid_df_train))    #下标从0开始
    uid_texts = uid_df_train.fenci.values
    uid_dictionary = corpora.Dictionary(uid_texts)
    uid_corpus = [uid_dictionary.doc2bow(text) for text in uid_texts]
    tfidf = models.TfidfModel(uid_corpus)
    uid_corpus_tfidf = tfidf[uid_corpus]
    lsi = models.LsiModel(uid_corpus_tfidf, id2word=uid_dictionary, num_topics=2)    # 主题个数
    uid_index = similarities.MatrixSimilarity(lsi[uid_corpus])
    for mid in uid_df_test.mid.values:
        uid_query = uid_df_test[uid_df_test.mid==mid].fenci.values[0]
        uid_query_bow = uid_dictionary.doc2bow(uid_query)
        uid_query_lsi = lsi[uid_query_bow]
        uid_sims = uid_index[uid_query_lsi]
        uid_sort_sims = sorted(enumerate(uid_sims), key=lambda item: -item[1])
        # 找到最相似的几条微博
        a = []
        for i in xrange(5):
            a.append(uid_sort_sims[i][0])
        a = uid_df_train.ix[a]
        uid_df_test[uid_df_test.mid==mid].fp = int(a.mean().f)
        uid_df_test[uid_df_test.mid==mid].cp = int(a.mean().c)
        uid_df_test[uid_df_test.mid==mid].lp = int(a.mean().l)
    print 'TestPredict4: ', Precision(uid_df_test)

# 法五测试，对一个用户试试，按f分段转换成分类问题
#def TestPredict5(uid_df_train, uid_df_test):
uid_df_train.index = range(len(uid_df_train))
label_train = [0]*len(uid_df_train)
uid_df_train['label'] = float(0.0)
uid_df_test['label'] = 0
uid_df_train[(uid_df_train.f > 10) & (uid_df_train.f <= 100)].label = 1
uid_df_train[(uid_df_train.f > 100) & (uid_df_train.f <= 500)].label = 2
uid_df_train[(uid_df_train.f > 500) & (uid_df_train.f <= 1000)].label = 3
uid_df_train[uid_df_train.f > 1000].label = 4
def SetLabel(f):
    if f <= 10 :
        return 0
    elif f > 10 and f <= 100:
        return 1
    elif f > 100 and f <= 1000:
        return 2
    else:
        return 3
uid_df_train.label = uid_df_train.f.map(lambda f: SetLabel(f))

# 写出预测数据到文件
def Output(pred):
    file = open('weibo_result_data.txt', 'w')
    for ind in xrange(len(pred)):
#        file.write(pred['uid'][ind]+'\t'+pred['mid'][ind]+'\t'+str(int(round(pred['fp'][ind])))+','\
#                +str(int(round(pred['cp'][ind])))+','+str(int(round(pred['lp'][ind])))+'\n')
        file.write(pred['uid'][ind]+'\t'+pred['mid'][ind]+'\t'+str(pred['fp'][ind])+','\
                +str(pred['cp'][ind])+','+str(pred['lp'][ind])+'\n')
    file.close()


df = LoadTrainData()
df_pre = LoadPredictData()
df['fenci'] = pd.Series(LoadFenCi())

# 训练样本和测试样本
df_train = df[df.time <= pd.Timestamp('2015-06-30')]    #前5个月作训练
df_test = df[df.time > pd.Timestamp('2015-06-30')]    #最后一个月作测试

df_train_uni = df_train.groupby('uid').mean().sort('f', ascending=False)
df_test_uni = df_test.groupby('uid').mean().sort('f', ascending=False)


#uid = df_train_uni.index[2]
uid = 'e88330514585dc40b7cb8f48c0e0ea2a'    # 错误率最高的用户
uid_df_train = df_train[df_train.uid == uid]
uid_df_test = df_test[df_test.uid == uid]
uid_df = df[df.uid == uid]
# Predict1 0.016016981860285603
# Predict2 0.0
# Predict3 0.11694326514859128(5近邻,3近邻)，0.0661906599768（最近邻）

# 用一部分数据测试一下Predict3的效果
#df_train_part = df_train.head(10000)    # 简单地取前10000条数据
#df_test_part = df_test.head(10000)
Predict1(df_test)
p1 = Precision(df_test)
Predict2(df_train, df_test)
p2 = Precision(df_test)
Predict3(df_train, df_test)
p3 = Precision(df_test)
# 正确率比较# 1000条 10000条
print 'Predict1:', p1
print 'Predict2:', p2# 0.370 0.339
print 'Predict3:', p3# 0.364 0.335


def GetWordCounts(uid_mid):
    line = uid_df_train[uid_df_train.mid==uid_mid].fenci.values[0]
    wc = {}
    for word in line:
        wc.setdefault(word, 0)
        wc[word] += 1
    return uid_mid, wc

apcount = {}
wordcounts = {}
for uid_mid in uid_df_train.mid.values:
    title, wc = GetWordCounts(uid_mid)
    wordcounts[title] = wc
    for word, count in wc.items():
        apcount.setdefault(word, 0)
        if count>1:
            apcount[word] += 1

wordlist = []
for w, bc in apcount.items():
    frac = float(bc)/len(uid_df_train)
    if frac>0.1 and frac<0.5: wordlist.append(w)

uid_df_train['feature'] = uid_df_train.mid.map(wordcounts)

