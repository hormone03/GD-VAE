import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.cuda
from pprint import pprint, pformat
import pickle
import argparse
import os
import math
import matplotlib.pyplot as plt
from collections import Counter

from pytorch_model import ProdLDA

import pandas as pd
from tqdm import tqdm
from itertools import chain
import nltk


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-t', '--num_topic',        type=int,   default=50)
parser.add_argument('-b', '--batch_size',       type=int,   default=200)
parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
parser.add_argument('-r', '--learning_rate',    type=float, default=0.1)
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
parser.add_argument('-e', '--num_epoch',        type=int,   default=80)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)
parser.add_argument('-v', '--variance',         type=float, default=0.995)
parser.add_argument('-alph', '--alpha',         type=float, default=0.02)
parser.add_argument('-bet', '--beta',           type=float, default=0.02)
parser.add_argument('--start',                  action='store_true')
parser.add_argument('--nogpu',                  action='store_true')

args = parser.parse_args()

def count_word_combination(dataset,combination):
    count = 0
    w1,w2 = combination
    for data in dataset:
        w1_found=False
        w2_found=False
        for word_id, freq in data.items():
            if not w1_found and word_id==w1:
                w1_found=True
            elif not w2_found and word_id==w2:
                w2_found=True
            if w1_found and w2_found:
                count+=1
                break
    return count     


def count_word(dataset,word):
    count=0
    for data in dataset:
        for word_id, freq in data.items():
            if word_id==word:
                count+=1
                break
    return count 


def setData_Coherence(url):
    data_te = np.load(url, allow_pickle=True,  encoding='latin1')
    dataTEST = []
    count = []
    for data in data_te:
        #d = {x:data.count(x) for x in data}
        count.append(len(data))
        d = Counter(data)
        dataTEST.append(d)
    return dataTEST, count


def topic_coherence(dataset,beta, n_top_words=10):
    word_counts={}
    word_combination_counts={}
    length = len(dataset)
    coherence_sum=0.0
    coherence_count=0
    topic_coherence_sum=0.0
  
    for i in range(len(beta)):
        top_words = [j
            for j in beta[i].argsort()[:-n_top_words - 1:-1]]
        topic_coherence = 0
        topic_coherence_count=0.0
        for i,word in enumerate(top_words):
            if word not in word_counts:
                count = count_word(dataset,word)
                word_counts[word]=count
            for j in range(i):
                word2 = top_words[j]
                combination = (word,word2)
                if combination not in word_combination_counts:
                    count = count_word_combination(dataset,combination)
                    word_combination_counts[combination]=count
                wc1 = word_counts[word]/float(length)
                wc2 = word_counts[word2]/float(length)
                cc = (word_combination_counts[combination])/float(length)
                if cc>0:
                    coherence = math.log(cc/float(wc1*wc2))/(-math.log(cc))
                    topic_coherence+=coherence
                    coherence_sum+=coherence
                coherence_count+=1
                topic_coherence_count+=1
        topic_coherence_sum+=topic_coherence/float(topic_coherence_count)
    return coherence_sum/float(coherence_count),topic_coherence_sum/float(len(beta))

    
def diversity(beta, n_top_words):
    if beta is None:
        return 0
    if n_top_words > len(beta[0]):
        raise Exception('Words in topics are less than ' + str(n_top_words))
    else:
        unique_words = set()
        for topic in beta:
            unique_words = unique_words.union(set(topic[:n_top_words]))
        diver = len(unique_words) / (n_top_words * args.num_topic)
        return diver


def Redundancy(input_data, feature_names, n_top_words):

    try:
        assert 5 <= n_top_words <= 15
        
    except AssertionError:
        
        "Invalid Value for n (int) [5,15]"
        
        raise
        
    ngram = [0]*len(input_data)
                          
    for i in tqdm(range(len(input_data)), desc = 'Get the Redundancy'):
        line = " ".join([feature_names[j] 
                        for j in input_data[i].argsort()[:-n_top_words - 1:-1]]) 
                            
        ngram[i] = [0]*len(input_data[i])
        for j in range(len(input_data[i])):
            if input_data[i] !='':
                ngram[i][j] = list(nltk.ngrams(line.split(),10))
    
    list_ngrams_per_doc = [list(chain(*ngram[i])) for i in range(len(ngram))]
    
    # calculate the Redundancy
    redundancy = [pd.DataFrame({"Ten_grams":list_ngrams_per_doc[i]}).\
                  value_counts().\
                  loc[lambda x: x>1].\
                  sum()/\
                  len(list_ngrams_per_doc[i]) for i in range(len(list_ngrams_per_doc))]
    
    return redundancy


# default to use GPU, but have to check if GPU exists
if not args.nogpu:
    if torch.cuda.device_count() == 0:
        args.nogpu = True

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)


def make_data():
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size, dataset_te
    dataset_tr = 'data/20news_clean/train.txt.npy'
    data_tr = np.load(dataset_tr, allow_pickle=True,  encoding='latin1')
    dataset_te = 'data/20news_clean/test.txt.npy'
    data_te = np.load(dataset_te, allow_pickle=True,  encoding='latin1')
    vocab = 'data/20news_clean/vocab.pkl'
    vocab = pickle.load(open(vocab,'rb'))
    vocab_size=len(vocab) 
    print(vocab_size)
    #--------------convert to one-hot representation------------------
    print('Converting data to one-hot representation')
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print('Data Loaded')
    print('Dim Training Data',data_tr.shape)
    print('Dim Test Data',data_te.shape)
    #--------------make tensor datasets-------------------------------
    data_tr=np.vstack(data_tr).astype(np.float)
    data_te=np.vstack(data_te).astype(np.float)
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    
    if not args.nogpu:
        tensor_tr = tensor_tr.cuda()
        tensor_te = tensor_te.cuda()

def make_model():
    global model
    net_arch = args 
    net_arch.num_input = data_tr.shape[1]
    model = ProdLDA(net_arch)
    if not args.nogpu:
        model = model.cuda()

def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def train():
    for epoch in range(args.num_epoch):
        all_indices = torch.randperm(tensor_tr.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            if not args.nogpu: batch_indices = batch_indices.cuda()
            input = Variable(tensor_tr[batch_indices])
            recon, loss = model(input, compute_loss=True)
            # optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # report
            loss_epoch += loss.data #loss.data[0]    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))

associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh', 
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'politics': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}
def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.items():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

def print_top_words(beta, feature_names, n_top_words=10): 
    print('---------------Printing the Topics------------------')

    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]]) 

        topics = identify_topic_in_line(line)
        print('|'.join(topics))
        print('     {}'.format(line))
    print('---------------End of Topics------------------')

def print_perp(model):
    cost=[]
    model.eval()                        # switch to testing mode
    input = Variable(tensor_te)
    recon, loss = model(input, compute_loss=True, avg_loss=False)
    loss = loss.data
    counts = tensor_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', math.exp(avg))



if __name__=='__main__' :
    make_data()
    make_model()
    make_optimizer()
    train()
    emb = model.decoder.weight.data.cpu().numpy().T
    
    print_perp(model)
    print('############## Embedding ##################')
    print(emb[0])
    print('calculate topic coherence (might take a few minutes)')
    dataTEST, count = setData_Coherence(dataset_te)
    h, topicCoh = topic_coherence(dataTEST,emb, n_top_words=10)
    print('topic coherence',str(topicCoh))
    diver_sity = diversity(emb,50)
    print('topic diversity', str(diver_sity))  
    diver_sity = diversity(emb,50)
    Redundancy_ = Redundancy(emb, list(zip(*sorted(vocab.items(), key=lambda x:x[1])))[0], n_top_words=10)  
    print('topic Redundancy', str(Redundancy_))


