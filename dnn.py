# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 13:21:18 2018

@author: wangyizhe
"""
import random
import numpy as np
import tensorflow as tf
from utils import is_valid_icd10
import pandas as pd
#from sklearn.model_selection import train_test_split
#from collections import Counter
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   #指定使用GPU序号，从0开始，多个GPU逗号分隔，-1表示禁用GPU。

SEED = 2018
random.seed(SEED)
np.random.seed(SEED)

'''
def standardization(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    arr = (arr-mean)/std
    return arr, std, mean
'''
def preload_W2V(path='data/it2vec'):
    preload_dict = {}
    print('Start reading pretrained vector from directory: ' + path)
    with open('data/it2vec') as f:
        # First line
        l = f.readline().strip().split()
        dim = int(l[1])
        num_vecs = int(l[0])
        for line in f.readlines():
        
            l = line.strip().split()
            st = l[0]
            preload_dict[st] = np.array(l[1:])
    print('Loaded {} vectors with dim = {}'.format(num_vecs, dim))
    return preload_dict, num_vecs, dim

def generate_embedding(xm_dict, emb_size = 100,use_pretrained=False):
    embeddings = np.random.uniform(-1,1,size=(len(xm_dict), emb_size))
    if use_pretrained:
        preload_dict, num_vecs, dim = preload_W2V()
        if (dim != emb_size):
            print('Embedding size does not match, will use random initialization')
        else:
            cnt = 0
            for word in xm_dict:
                if word in preload_dict:
                    idx = xm_dict[word]
                    embeddings[idx] = preload_dict[word]
                    cnt+=1
            print('Mapped ' + str(cnt) + ' Vectors')
    return embeddings

def val2idx(words, labels, label_dict, xm_dict, max_window_size = 300):
    wc = len(words)
    nc = len(label_dict)
    
    x = np.zeros([wc, max_window_size], np.int32)
    y = np.zeros([wc, nc], np.int32)
    
    i = 0
    for line in words:
        idx = label_dict[labels[i]]
        y[i][idx] = 1
        j = 0
        for word in line.split(','):
            if word in xm_dict:
                x[i][j] = xm_dict[word]
                j+=1
            else:
                print('Error! New Program Encountered: ' + word)
            if j > max_window_size:
                print('Reached Maximum, Consider increasing bandwith')
                break
        i+=1
    return x,y

def data_prep(df):
    df = df.copy()
    columns = ["CYZDDM","HZNL","JGMC","JGDJ","CYLY","RYLB","RYZT","JGLB","YLLB","HZXB","XMDM"]
    df = df[columns]
    for col in columns:
        df = df[df[col].notnull()]
    df = df[df['HZNL'].isin(range(0, 130))]
    df = df[df['CYZDDM'].map(is_valid_icd10)]
    df['HZXB'] = df['HZXB'].map(lambda x: 0 if x.strip() == '男' else 1)
    df['JGDJ'] = df['JGDJ'] - 1
    
    dictionary = {}
    j=0
    dictionary['RYLB'] = {item: i+j for i, item in enumerate(set(df['RYLB']))}
    j+=len(dictionary['RYLB'])
    dictionary['RYZT'] = {item: i+j for i, item in enumerate(set(df['RYZT']))}
    j+=len(dictionary['RYZT'])
    dictionary['JGMC'] = {item: i+j for i, item in enumerate(set(df['JGMC']))}
    j+=len(dictionary['JGMC'])
    dictionary['JGLB'] = {item: i+j for i, item in enumerate(set(df['JGLB']))}
    j+=len(dictionary['JGLB'])
    dictionary['YLLB'] = {item: i+j for i, item in enumerate(set(df['YLLB']))}
    j+=len(dictionary['YLLB'])
    dictionary['CYLY'] = {item: i+j for i, item in enumerate(set(df['CYLY']))}
    j+=len(dictionary['CYLY'])
    df['RYLB'] = df['RYLB'].map(dictionary['RYLB'])
    df['RYZT'] = df['RYZT'].map(dictionary['RYZT'])
    df['JGMC'] = df['JGMC'].map(dictionary['JGMC'])
    df['JGLB'] = df['JGLB'].map(dictionary['JGLB'])
    df['YLLB'] = df['YLLB'].map(dictionary['YLLB'])
    df['CYLY'] = df['CYLY'].map(dictionary['CYLY'])
    return df, j

def build_dataset(read_file='data/wuxi_zc_18_7_4.csv', max_window_size = 300):
    df = pd.read_csv(read_file, encoding='gbk')
    print('Processing input file...')
    df, side_size = data_prep(df)
    print('Done')
    #dictionaries
    xm_dict = {}
    label_dict = {}
    
    # reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    words = df['XMDM'].values
    labels = list(df['CYZDDM'].map(lambda x: x[:3]))
    # 年龄性别
    ages = np.array(df['HZNL'],dtype=np.float64)/130
    genders = np.array(df['HZXB'],dtype=np.float64)
    # 其他外部信息
    side_info = np.array(df[["JGMC","JGDJ","CYLY","RYLB","RYZT","YLLB"]])
    # 字典构建
    label_cnt = 0
    for zd in labels:
        if zd not in label_dict:
            label_dict[zd] = label_cnt
            label_cnt+=1    
    xm_cnt = 0
    for line in words:
        for word in line.split(','):
            if word not in xm_dict:
                xm_dict[word] = xm_cnt
                xm_cnt+=1
    # Mapping Index
    wc = np.array(df['XMDM'].map(lambda x: len(x.split(','))))
    m = np.zeros([len(wc), max_window_size], np.int32)
    idx = 0
    for i in wc:
        m[idx][:i] = 1
        idx+=1
    words,labels = val2idx(words,labels,label_dict,xm_dict)
    return words, labels, ages.reshape([-1,1]), genders.reshape([-1,1]), side_info, side_size, m, wc.reshape([-1,1]), len(label_dict), xm_dict
#========================

class dnn(object):
    def generate_batch(self, pos):
    # CBOW w/ bagSize = num of subjects
    # if pos+batch_size >= len(words):
    #    batch_size = len(words) - pos
        target = self.data['idxes'][pos:pos+self.batch_size]
        x = self.data['x'][target]
        y = self.data['y'][target]
        mask = self.data['mask'][target]
        word_cnt = self.data['wc'][target]
        a = self.data['side1'][target]
        g = self.data['side2'][target]
        s = self.data['side_info'][target]
        #word_cnt = np.count_nonzero(mask, axis=1)
        return x, mask.reshape(self.batch_size, self.max_window_size, 1), word_cnt, y, a, g, s
        #return x, word_cnt, y
        
        
    def __init__(self, data, n_classes, vocab_size, side_size, side_window_size, ratio = 0.9, max_window_size=300, epochs = 2000, emb_side_size = 30, embedding_size = 100, n_hidden_1 = 256, n_hidden_2 = 100, batch_size = 64, lr = 0.0001):
        
        # Hyper-params
        self.data = data
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.emb_size = embedding_size
        self.emb_side_size = emb_side_size
        self.vocab_size = vocab_size
        self.side_size = side_size
        self.learning_rate = lr
        self.epochs = epochs
        self.max_window_size = max_window_size
        self.side_window_size = side_window_size
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.ratio = ratio
        # 其他信息种类
        self.side_info = 2
        # 数据归一化
        #L = ['age', ]
        # create a session
        self._init_graph()
        self.sess = tf.Session(graph=self.graph)
        # self.sess = tf.InteractiveSession(graph=self.graph)
        
    def _init_graph(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(SEED)
            # Embedding Layer
            self.side_embeddings = tf.get_variable(name='side_info', shape=(self.side_size, self.emb_side_size), dtype=tf.float64, initializer = tf.random_normal_initializer(0,1))
            #self.embeddings = tf.get_variable(name='embedding',shape=(self.vocab_size, self.emb_size), dtype=tf.float32, initializer= tf.random_normal_initializer(0,1))
            self.embeddings = tf.get_variable(name='embedding', initializer= self.data['embeddings'])
            self.emb_mask = tf.placeholder(tf.float64, shape=(None, self.max_window_size, 1),name='emb_mask')
            self.word_num = tf.placeholder(tf.float64, shape=(None, 1),name='word_num')
            self.x = tf.placeholder(tf.int32, shape=(None, self.max_window_size),name='x')
            self.y = tf.placeholder(tf.int64, shape=(None, self.n_classes),name='y')
            self.keep_prob = tf.placeholder(tf.float64)
            # Side Info
            self.age = tf.placeholder(tf.float64,shape=(None,1))
            self.gender = tf.placeholder(tf.float64, shape=(None,1))
            self.x_side = tf.placeholder(tf.int32, shape=(None,self.side_window_size))
            # Input->Hidden
            #self.W1 = tf.get_variable('W1',[self.emb_size, self.n_hidden_1],initializer = tf.contrib.layers.xavier_initializer())
            self.W11 = tf.get_variable('W11',[self.emb_size+self.emb_side_size+self.side_info, self.n_hidden_1], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer())
            self.B11 = tf.get_variable('B11',[self.n_hidden_1], dtype=tf.float64, initializer = tf.zeros_initializer())
            self.W12 = tf.get_variable('W12',[self.n_hidden_1, self.n_hidden_2], dtype=tf.float64, initializer = tf.contrib.layers.xavier_initializer())
            self.B12 = tf.get_variable('B12',[self.n_hidden_2], dtype=tf.float64, initializer = tf.zeros_initializer())
            # Hidden->Softmax
            self.W2 = tf.get_variable(
                    name = 'W2',
                    shape=[self.n_hidden_2, self.n_classes],
                    dtype=tf.float64,
                    initializer=tf.contrib.layers.xavier_initializer())
            self.B2 = tf.get_variable('B2', initializer = tf.constant(0.1, dtype=tf.float64, shape=[self.n_classes]))
            
            # Model
            input_embedding = tf.nn.embedding_lookup(self.embeddings, self.x)
            side_embedding = tf.reduce_mean(tf.nn.embedding_lookup(self.side_embeddings, self.x_side),axis=1)
            self.H0 = tf.div(tf.reduce_sum(tf.multiply(input_embedding,self.emb_mask), 1),self.word_num)
            # concat w/ side info
            self.H0 = tf.concat([self.H0, side_embedding, self.age, self.gender],axis=1)
            #self.H0 = tf.div(tf.reduce_sum(input_embedding, 1), self.word_num)
            #self.Z1 = tf.add(tf.matmul(self.H0, self.W1), self.B1)
            #self.H1 = tf.nn.relu(self.Z1)
            #self.H1 = tf.nn.dropout(self.H1, self.keep_prob)
            #self.Z2 = tf.add( tf.matmul(self.H1,self.W2) ,self.B2)
            
            self.Z11 = tf.add(tf.matmul(self.H0, self.W11), self.B11)
            self.H11 = tf.nn.relu(self.Z11)
            self.H11 = tf.nn.dropout(self.H11, self.keep_prob)
            self.Z12 = tf.add(tf.matmul(self.H11, self.W12), self.B12)
            self.H12 = tf.nn.relu(self.Z12)
            self.H12 = tf.nn.dropout(self.H12, self.keep_prob)
            
            self.Z2 = tf.add( tf.matmul(self.H12,self.W2) ,self.B2)
            self.predictions = tf.argmax(self.Z2, 1)
            
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.y, logits = self.Z2)
            self.loss = tf.reduce_mean(losses)
            
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct_predictions, 'float'), name='accuracy')
    
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(losses)
            # initialization
            self.init_op = tf.global_variables_initializer()
            # create a saver 
            self.saver = tf.train.Saver([self.embeddings,self.side_embeddings,self.W11,self.W12,self.W2,self.B11,self.B12,self.B2])
        
        
    def fit(self,keep_prob):
        session = self.sess
        session.run(self.init_op)
        print('Initialzed')
        total_batches = len(self.data['idxes']) // self.batch_size
        train_batches = int(total_batches*self.ratio)
        test_batches = total_batches-train_batches
        print('Total Batches: '+ str(total_batches) + ' Training Batches: ' + str(train_batches))
        for epoch in range(self.epochs):
            epoch_cost = 0.
            # Shuffle trainning data every epoch
            
            #np.random.shuffle(self.data['idxes'])
            num_corrects = 0
            # training
            for i in range(train_batches):
                x,m,w,y,a,g,s = self.generate_batch(i*self.batch_size)
                feed_dict = {self.x_side:s, self.x:x, self.y:y, self.emb_mask:m, self.word_num:w, self.age:a, self.gender:g, self.keep_prob:keep_prob}
                #feed_dict = {self.x:x, self.y:y, self.word_num:w, self.keep_prob:keep_prob}
                op, l, acc = session.run([self.optimizer, self.loss, self.num_correct],feed_dict = feed_dict)
                epoch_cost += l
                num_corrects += acc
            print('epoch = {}, cost = {}, num_corrects = {}, accuracy = {}'.format(epoch, epoch_cost, num_corrects, num_corrects/(train_batches*self.batch_size)))
            
            # evaluation
            num_corrects = 0
            for i in range(test_batches):
                x,m,w,y,a,g,s = self.generate_batch((i+train_batches)*self.batch_size)
                feed_dict = {self.x_side:s, self.x:x, self.y:y, self.emb_mask:m, self.word_num:w, self.age:a, self.gender:g, self.keep_prob:1.0}
                #feed_dict = {self.x:x, self.y:y, self.word_num:w, self.keep_prob:keep_prob}
                l, acc = session.run([self.loss, self.num_correct],feed_dict = feed_dict)
                epoch_cost += l
                num_corrects += acc
            print('Validation epoch = {}, cost = {}, num_corrects = {}, accuray = {}'.format(epoch, epoch_cost, num_corrects, num_corrects/(test_batches*self.batch_size)))
        self.save()
        
    def predict(self, x):
        session = self.sess
        feed_dict = {}
        vec = session.run(self.Z2, feed_dict=feed_dict)
        return vec
    
    def save(self, path = 'model/model.ckpt'):
        save_path = self.saver.save(self.sess, path)
        return save_path
    
    def restore(self, path = 'model/model.ckpt'):
        self.saver.restore(self.sess, path)
        
def train():
    words, labels, ages, genders, side_info, side_size,mask, word_cnt, n_classes, xm_dict = build_dataset()
    #Train/Test Splitting
    idxes = np.array(range(len(words)))
    np.random.shuffle(idxes)
    data = {}
    data['idxes'] = idxes
    data['x'] = words
    data['y'] = labels
    data['side1'] = ages
    data['side2'] = genders
    data['side_info'] = side_info
    data['mask'] = mask
    data['wc'] = word_cnt
    # Using preload weights
    data['embeddings'] = generate_embedding(xm_dict, use_pretrained=True)
    
    #words, words_val, labels, labels_val, ages, ages_val, genders, genders_val = train_test_split(words, labels, ages, genders, test_size = 0.01, random_state = 42)
    #data['idxes'] = train_test_split(idxes, test_size = 0.01, random_state = 42)
    print("Class Num: ", n_classes)
    mydnn = dnn(data, n_classes, len(xm_dict), side_size, side_info.shape[1])
    mydnn.fit(0.5)

if __name__ == '__main__':
    train()