import numpy as np
import pandas as pd
import datetime
import sys
from glob import glob
from collections import Counter
import gensim
from scipy import spatial
import multiprocessing
from scipy import stats
import statsmodels.formula.api as smf
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import matplotlib.pyplot as plt
# from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush()

TaggededDocument = gensim.models.doc2vec.TaggedDocument

column_list = 'user_id,access_mode_id,logic_area_name,lac,ci,longtitude,latitude,busi_name,busi_type_name,\
app_name,app_type_name,start_time,up_pack,down_pack,up_flow,down_flow,site_name,site_channel,cont_app_id,\
cont_classify_id,cont_type_id,acce_url'.split(',')


from urllib import parse
 
# clean url
def urlclean(url):
    try:
        url = parse.urlparse(url).hostname
        if url.replace('.','').isdigit(): return 'none'
        else:
            if len(url.split('.')) >=2 :
                if url[-6:]=='com.cn': return '.'.join(url.split('.')[-3:])
                return '.'.join(url.split('.')[-2:])
    except:
        return 'none'

files=glob('./allusers/*.csv')
x_train=[]
freq=[]
for i,f in enumerate(files):
    # data=pd.read_csv(f,names=column_list)
    # data['app_name1']=data.apply(lambda x: urlclean(x.acce_url) if x.app_name=='其他' else x.app_name,axis=1)
    # data['app_type_name1']=data.apply(lambda x: x.app_type_name if x.app_type_name!='-9' else x.busi_type_name,axis=1)
    # data=data.sort_values(by=['start_time'],ascending=True)
    # data=data[data['app_name1']!='none']
    # data=data[['app_name1','app_type_name1']]
    # data['app_type_name1']=data['app_type_name1'].replace(np.nan,'nan')
    a=f.split('user_')[-1].split('.')[0]
    # freq.append([a,Counter(data['app_type_name1'].values)])
    # data.to_csv('./allusers1/'+str(a)+'.csv',index=False)
    data=pd.read_csv('./allusers1/'+str(a)+'.csv')
    data=data.dropna(how='all')
    doc=[str(i) for i in list(data['app_name1'].values)]
    doc_type=[str(i) for i in list(data['app_type_name1'].values)]
    word_list=[]
    for j in range(len(doc)-1):
        if doc[j]!=doc[j+1]:
            word_list.extend([doc_type[j],doc[j]])
    document = TaggededDocument(word_list, tags=[i])
    x_train.append(document)


a=pd.DataFrame(freq)
a.to_csv('./freq_06_13.csv',index=False)

def train(x_train, size=300, epoch_num=1):
    model_dm = Doc2Vec(x_train, size = size, dbow_words=1, min_count=1)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('doc2vec_06_13')
 
    return model_dm

model_dm=train(x_train)

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def projection(one, two):
    # cosine similarity
    return (1 - spatial.distance.cosine(one, two))

def cosine2angle(i):
    return np.arccos(i) / np.pi * 180 #长度换角度


radius=[]
for j in range(len(x_train)):
    new_names=['位置','其他资讯','房产','招聘','搜索','支付','旅游','游戏','生活服务','电商购物',
                         '社交沟通','社区论坛','网页浏览','视频','软件工具','邮箱','阅读','音乐','nan','应用商店']
    d={}
    for m in range(len(new_names)):
        a1=[]
        for i in range((len(x_train[j].words)+1)//2):
            if x_train[j].words[2*i]==new_names[m]:
                a=(-projection(normalize(model_dm.docvecs[x_train[j].tags[0]]),normalize(model_dm[x_train[j].words[2*i+1]]))+1)/2
                a1.append(a)
                d[new_names[m]]=a1
        if new_names[m] not in d:
            d[new_names[m]]=1
    radius.append([x_train[j].tags[0],d])
d=pd.DataFrame(radius)
d.to_csv('./rg_06_13.csv',index=False)