import numpy as np
import pandas as pd
import datetime
import sys
import multiprocessing
from glob import glob
from collections import Counter
import gensim
from scipy import spatial
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
import matplotlib.pyplot as plt
# from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
import statsmodels.formula.api as smf
from scipy.stats import chisquare
from scipy import stats
pd.set_option('display.max_columns',None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

def flushPrint(s):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % s)
    sys.stdout.flush()

TaggededDocument = gensim.models.doc2vec.TaggedDocument

column_list = 'user_id,no,time_duration,time,book_id,type'.split(',')
book=pd.read_csv('./book_table.csv')
book=book[book['category'].notnull()]
diction=dict(book[['bookid','category']].values)

files=glob('./allusers/*.csv')
x_train=[]
freq=[]
for i,f in enumerate(files):
    data=pd.read_csv(f,names=column_list)
    data=data[['user_id','time_duration','book_id','time']]
    data=data.sort_values(by=['time'],ascending=True)
    data=data[data['book_id'].notnull()]
    data['cate']=data['book_id'].apply(lambda x: diction[x] if x in diction else np.nan)
    data['cate']=data['cate'].replace(np.nan,'nan')
    data['book_id']=data['book_id'].replace(np.nan,'nan')
    data=data.drop_duplicates()
    data=data[data['time_duration']!=0]
    data=data[~data['book_id'].str.contains('.txt')]
    a=f.split('/')[-1].split('.')[0]
    freq.append([a,dict(data.groupby(['cate'])['time_duration'].sum()),dict(Counter(data['cate'].values))])
    doc=list(data['book_id'].values)
    doc_type=list(data['cate'].values)
    word_list=[]
    for j in range(len(doc)-1):
        if doc[j]!=doc[j+1]:
            word_list.extend([doc_type[j],doc[j]])
    document = TaggededDocument(word_list, tags=[i])
    x_train.append(document)

a=pd.DataFrame(freq)
a.to_csv('./freq_07_05.csv',index=False)


def train(x_train, size=300, epoch_num=1):
    model_dm = Doc2Vec(x_train, size = size, dbow_words=1, min_count=1)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('doc2vec_07_05')
 
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
    new_names=['历史传记','原创女频','小说','计算机','社会科学','外文','亲子少儿','文学艺术','原创男频','生活','经济管理',
               '成功励志','科技','期刊','两性情感','nan']
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
d.to_csv('./rg_07_05.csv',index=False)


