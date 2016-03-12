# -*- coding: utf-8 -*-
"""
Created on Sat Aug 01 22:52:54 2015

@author: Administrator
"""

import numpy as np
import pandas as pd
import jieba
import string
import re

# 解决中文编码问题
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

# 读入训练数据
#df = pd.read_csv('weibo_train_data.txt', sep='\t', names=['uid', 'mid', 'time', 'f', 'c', 'l', 'co'])
#df[['f', 'c', 'l']] = df[['f', 'c', 'l']].astype(float)
#df.time = pd.to_datetime(df.time)    # 但是这种方法会导致丢失4000个点，不知道为什么
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

#s6 = s6.map(lambda x: x.translate(None, string.punctuation+' '+'\n'+string.digits))
#s6 = s6.map(lambda x: jieba.lcut(x))

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

  
# 按比例随机选取训练和测试样本
def SelectTrainTest(df):
    num_of_train = round(len(df)*0.9)
    #num_of_test = round(len(df) - num_of_train)
    random_index = np.random.choice(df.index, len(df), replace=False)
    train_df = df.iloc[random_index[:num_of_train]]
    test_df = df.iloc[random_index[num_of_train:]]
    return train_df, test_df
     
     
# 1.看预测数据中的uid，如果在训练数据中存在，则fcl就取改用户所有fcl的均值；如果不存在，就赋010。
def Predict1(train_df, test_df):
    test_df['fp'] = 0
    test_df['cp'] = 1
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
#    test_df['fp'] = test_df['uid'].map(f_dic)
#    test_df['cp'] = test_df['uid'].map(c_dic)
#    test_df['lp'] = test_df['uid'].map(l_dic)
    #test_df['cp'] = round(test_df['cp'])
    #test_df['lp'] = round(test_df['lp'])
    # 对于训练集中不存在的用户，取结果的中位数
    #test_df.astype(object).fillna(test_df.median())
    return test_df 
    #pdf['f'] = pdf['uid'].map(lambda x: media[x.values[0]]['f'] if x.values[0] in media.index else 0)
    
    
# 计算整体准确率
def Precision(df):
    df['prei'] = 1-0.5*abs(df.fp-df.f)/(df.f+5)-0.25*abs(df.cp-df.c)/(df.c+3)-0.25*abs(df.lp-df.l)/(df.l+3)
    df['sgn'] = 0
    df.loc[df.prei-0.8>0, 'sgn'] = 1
    df['coui'] = df.f + df.c + df.l
    df.loc[df.coui>100, 'coui'] = 100
    p = ((df['coui']+1)*df['sgn']).sum() / (df['coui']+1).sum()
    return p

# 写出预测数据到文件
def Output(pred):
    file = open('weibo_result_data.txt', 'w')
    for ind in xrange(len(pred)):
#        file.write(pred['uid'][ind]+'\t'+pred['mid'][ind]+'\t'+str(int(round(pred['fp'][ind])))+','\
#                +str(int(round(pred['cp'][ind])))+','+str(int(round(pred['lp'][ind])))+'\n')
        file.write(pred['uid'][ind]+'\t'+pred['mid'][ind]+'\t'+str(pred['fp'][ind])+','\
                +str(pred['cp'][ind])+','+str(pred['lp'][ind])+'\n')
    file.close()

# 用jieba分词（已经分好，保存在"fenci.txt"文件中）
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
    
    f = open('fenci.txt', 'w')
    for line in texts:
        k = ','.join(line)
        f.write(k+'\n')
    f.close()
    
# 读入分词结果
def LoadFenCi():
    texts = []
    f = open('fenci.txt', 'r')
    for line in f:
        texts.append(line.strip('\n').split(','))
    f.close()
    return texts

#train_df, test_df = SelectTrainTest(df)
train_df = df[df.time <= pd.Timestamp('2014-12-1')]    #前5个月作训练
test_df = df[df.time > pd.Timestamp('2014-12-1')]    #最后一个月作测试
result_df = Predict1(train_df, test_df)
p = Precision(result_df)
print p
texts = LoadFenCi()


#  对一个用户的新微博，尝试找到与其最相近的几条老微博
df['fenci'] = texts
train_df = df[df.time <= pd.Timestamp('2014-12-1')]    #前5个月作训练
test_df = df[df.time > pd.Timestamp('2014-12-1')]    #最后一个月作测试
#uid_sort_by_f = df.groupby('uid').sum().sort('f', ascending=False)
#uid = uid_sort_by_f.index[0]
#uid_train = train_df[train_df.uid==uid]    #该用户的老微博
#uid_train.index = xrange(len(uid_train))    #下标从0开始
#uid_test = test_df[test_df.uid==uid]    #该用户的新微博
## 先算一下如果新微博全赋中位数是的Precision
##uid_test['fp'] = uid_train.median().f
##uid_test['cp'] = uid_train.median().c
##uid_test['lp'] = uid_train.median().l
##p_median = Precision(uid_test)    #此时的Precision只有0，还不如全赋0的准确率高！！！！！
## 计算赋最相似微博的fcl的Precision
#from gensim import corpora, models, similarities
#import logging
#logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
#uid_texts = uid_train.fenci.values
#uid_dictionary = corpora.Dictionary(uid_texts)
#uid_corpus = [uid_dictionary.doc2bow(text) for text in uid_texts]
#tfidf = models.TfidfModel(uid_corpus)
#uid_corpus_tfidf = tfidf[uid_corpus]
#lsi = models.LsiModel(uid_corpus_tfidf, id2word=uid_dictionary, num_topics=10)
#uid_index = similarities.MatrixSimilarity(lsi[uid_corpus])
#fp = []
#cp = []
#lp = []
#def Calcfcl(uid_query):
#    #这个函数是对每条记录的微博分词进行处理，不太方便
##    uid_query = uid_test.fenci.values[0]
#    uid_query_bow = uid_dictionary.doc2bow(uid_query)
#    uid_query_lsi = lsi[uid_query_bow]
#    uid_sims = uid_index[uid_query_lsi]
#    uid_sort_sims = sorted(enumerate(uid_sims), key=lambda item: -item[1])
#    # 找到最相似的几条微博
#    a = []
#    for i in xrange(3):
#        a.append(uid_sort_sims[i][0])
#    a = uid_train.ix[a]
#    # 取最相似的微博的fcl中位数
##    fp.append(int(a.median().f))
##    cp.append(int(a.median().c))
##    lp.append(int(a.median().l))
#    # 取最相似的微博的fcl均值
#    fp.append(int(a.mean().f))
#    cp.append(int(a.mean().c))
#    lp.append(int(a.mean().l))
#
#meiyong = uid_test.fenci.map(lambda x: Calcfcl(x))
#uid_test.fp = pd.Series(fp, index=uid_test.index)
#uid_test.cp = pd.Series(cp, index=uid_test.index)
#uid_test.lp = pd.Series(lp, index=uid_test.index)
#print Precision(uid_test)    #Precision只有0.079


from gensim import corpora, models, similarities
import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)
#def Predict2(train_df, test_df):
    #写一个可以直接对测试数据中某一用户所有微博进行处理的函数
    #uniId_train = train_df.drop_duplicates(['uid']).uid.values    #训练样本单一用户
uniId_sum_f = train_df.groupby(['uid']).sum().sort_index(by='f', ascending=False)    #训练样本每个用户的f和并按降序排列
#uniId_train_100 = uniId_sum_f[uniId_sum_f>=100].index.values    #训练样本中总f大于100的用户
uniId_train_100 = uniId_sum_f.head(10).index.values    #取f最大的前10个用户ID
uniId_test = test_df.drop_duplicates(['uid']).uid.values    #测试样本单一用户
for uid in uniId_test:
    if uid in uniId_train_100:
        uid_train = train_df[train_df.uid==uid]    #该用户的老微博
        uid_train.index = xrange(len(uid_train))    #下标从0开始
        uid_test = test_df[test_df.uid==uid]    #该用户的新微博
        uid_texts = uid_train.fenci.values
        uid_dictionary = corpora.Dictionary(uid_texts)
        uid_corpus = [uid_dictionary.doc2bow(text) for text in uid_texts]
        tfidf = models.TfidfModel(uid_corpus)
        uid_corpus_tfidf = tfidf[uid_corpus]
        lsi = models.LsiModel(uid_corpus_tfidf, id2word=uid_dictionary, num_topics=10)
        uid_index = similarities.MatrixSimilarity(lsi[uid_corpus])
        for mid in uid_test.mid.values:
            uid_query = uid_test[uid_test.mid==mid].fenci.values[0]
            uid_query_bow = uid_dictionary.doc2bow(uid_query)
            uid_query_lsi = lsi[uid_query_bow]
            uid_sims = uid_index[uid_query_lsi]
            uid_sort_sims = sorted(enumerate(uid_sims), key=lambda item: -item[1])
            # 找到最相似的几条微博
            a = []
            for i in xrange(1):
                a.append(uid_sort_sims[i][0])
            a = uid_train.ix[a]
            result_df[result_df.mid==mid].fp = int(a.mean().f)
            result_df[result_df.mid==mid].cp = int(a.mean().c)
            result_df[result_df.mid==mid].lp = int(a.mean().l)



#p = np.array([0.0]*5)
#for i in xrange(5):
#    train_df, test_df = SelectTrainTest(df)
#    result_df = Predict(train_df, test_df)
#    p[i] = Precision(result_df)
#print p.mean()
#predict_df = LoadPredictData()
#result_df1 = Predict(df, predict_df)
#Output(result_df1)



'''
# 计算同一unique ID的所有微博转发、评论、点赞数的和
uni_id_sum = df.groupby('uid').sum()
print df[df['uid']==uni_id_sum.index[0]]

# 先提取链接？去除标点符号？用不用找@出的人？
line = df['co'][1].translate(None, string.punctuation+' '+'\n'+string.digits)
seg = jieba.lcut(line)
token = "，。！？：；的了".decode('utf-8')
fliter_seg = [x for x in seg if x not in token]
#line = line.translate(None, string.punctuation+' '+"，。！？：；".decode('utf-8'))


# 用结巴分词
seg_list = jieba.cut(df['co'][0], cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))
#  直接得到list
a = jieba.lcut(df['co'][0])
b = pd.Series(a)

#df['fp'] = 0
#df['cp'] = 0
#df['lp'] = 1

#print Precision(df)
'''

'''

   
# 初始化fcl
predict = LoadPredictData()
predict['f'] = 0
predict['c'] = 1
predict['l'] = 0


# 1.看预测数据中的uid，如果在训练数据中存在，则fcl就取改用户所有fcl的均值；如果不存在，就赋010。
uid_list = a.uid.values
pred_list = predict.uid.values
a['ave_f'] = a['uid'].map(lambda x: int(round(c.loc[x]['f'] / b[x])))
a['ave_c'] = a['uid'].map(lambda x: int(round(c.loc[x]['c'] / b[x])))
a['ave_l'] = a['uid'].map(lambda x: int(round(c.loc[x]['l'] / b[x])))
def pre_f(uid):
    if uid in uid_list:
        return round(a[a.uid == uid]['ave_f'])
predict['f'] = predict.apply(lambda row: a[a['uid']==row['uid']]['ave_f'] if row['uid'] in uid_list, axis=1)
for uid in predict.uid.values:
    if uid in uid_list:
        predict.loc[uid]['f'] = round(c.loc[uid]['f'] / b[uid])
        predict.loc[uid]['c'] = round(c.loc[uid]['c'] / b[uid])
        predict.loc[uid]['l'] = round(c.loc[uid]['l'] / b[uid])
f = []
c = []
l = []
for i in xrange(len(predict)):
    id = predict.ix[i].uid
    if id in uid_list:
        f.append(a[a['uid']==id]['ave_f'])
        c.append(a[a['uid']==id]['ave_c'])
        l.append(a[a['uid']==id]['ave_l'])
    else:
        f.append(0)
        c.append(1)
        l.append(0)

dic = a.set_index('uid')[['ave_f', 'ave_c', 'ave_l']].to_dict()
f = np.array([0]*len(predict))
c = np.array([1]*len(predict))
l = np.array([0]*len(predict))
for i in xrange(len(predict)):
    uid = pred_list[i]
    if uid in uid_list:
        f[i] = dic['ave_f'][uid]
        c[i] = dic['ave_c'][uid]
        l[i] = dic['ave_l'][uid]




Output(pred)



# 判断前5条微博内容是否包含“了”字
#print df['content'][:5].str.contains('了')

# 三种数据分布情况三维可视化
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(s3.values, s4.values, s5.values)
#ax.set_xlabel('forward_count')
#ax.set_ylabel('comment_count')
#ax.set_zlabel('like_count')
#plt.show()

# 去除最后30天的微博
df.time = pd.to_datetime(df.time)
df1 = df[df.time<'12-1-2014']

# 提取特征
wailian = re.compile(r"http://")
huati = re.compile(r"#(.+)#")
at = re.compile(r"@(.+) ")
df1['wailian'] = df1.co.map(lambda x: float(len(wailian.findall(x))))
df1['huati'] = df1.co.map(lambda x: float(len(huati.findall(x))))
df1['at'] = df1.co.map(lambda x: float(len(at.findall(x))))
df1['shijian'] = df1.time-pd.Timestamp('2014-07-01')
df1['shijian'] = df1['shijian'].astype('timedelta64[D]')

# 做回归
rand_idx = np.random.choice(df1.index, 500000, replace=False)
df2 = df1.ix[rand_idx]
x = df1[['wailian', 'huati', 'at', 'shijian']].values
y = list(df1['f'].values) 
from sklearn.cross_validation import train_test_split
#train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1, random_state=12345)
#from sklearn import linear_model
#regr = linear_model.LogisticRegression()
#regr = regr.fit(train_x, train_y)
#result = regr.predict(test_x)

# 去除重复
#a = df.drop_duplicates(['uid'])
a = df.drop_duplicates('uid')[['uid', 'time', 'f', 'c', 'l', 'co']]
b = df.groupby('uid').size()
c = df.groupby('uid').sum()
a['wailian'] = a.co.map(lambda x: float(len(wailian.findall(x))))
a['huati'] = a.co.map(lambda x: float(len(huati.findall(x))))
a['at'] = a.co.map(lambda x: float(len(at.findall(x))))
a['shijian'] = a.time-pd.Timestamp('2014-07-01')
a['shijian'] = a['shijian'].astype('timedelta64[D]')
a['num'] = b.values
a['ave'] = a.num/183.0
a[['f', 'c', 'l']] = c.values
a['meanf'] = a['f'] / a['num']
# 按发微博总量从大到小排序
a = a.sort(['num'], ascending=False)
# 按f从大到小排列
a = a.sort(['f'], ascending=False)
# 假设正常每天发10条，则一共在1800条，近似2000条左右
# 看看2000条左右都是多少条
sample = a.loc[(a.num<2100) & (a.num>1900)]
print sample
# 2000左右的1958条和2029条，看看他们的内容本别是什么(按转发数排序)
b2047 = df[df.uid == a[a.num==2047].index[0]].sort('f', ascending=False) #新闻
b2029 = df[df.uid == a[a.num==2029].index[0]].sort('f', ascending=False) #招聘广告
b1958 = df[df.uid == a[a.num==1958].index[0]].sort('f', ascending=False) #新闻
b1941 = df[df.uid == a[a.num==1941].index[0]].sort('f', ascending=False) #招聘广告

# 构建特征

# 分交叉验证数据

def Output(predict):
    file = open('weibo_result_data.txt', 'w')
    for ind in xrange(len(predict)):
        file.write(predict['uid'][ind]+'\t'+predict['mid'][ind]+'\t'+str(f[ind])+','\
                +str(c[ind])+','+str(l[ind])+'\n')
    file.close()


Output(predict)
'''