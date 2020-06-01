from utils import Corpus,  batchify, word_dict
import argparse

from model_loc import *
import torch.nn as nn
import numpy as np
import torch
'''
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=64)
args = parser.parse_args()

net = torch.load('./data/out/subj_big.pkl')
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'
device = torch.device(device)


corpus = Corpus('subj', test=True)

sum_all = 0
sum_count = 0
A_E = 1
for e in range(A_E):
    i = 0
    b = 0
    A_B = int(len(corpus.xs)//args.batch_size)
    
    while i >= 0:
        x, y = batchify(corpus, i, args.batch_size)
        if x is None:
            i = 1
            break
        x, y = x.to(device), y.to(device)
        out = net(x)
        sum_all += (1-(y^torch.argmax(out,dim=-1))).sum().to(torch.device('cpu'))
        sum_count += y.shape[0]
        i += args.batch_size
        b += 1
        if (b % 50 == 0):
            print('E: {}/{} | B: {}/{}'.format(e, A_E, b, A_B))

print(sum_all,sum_count)


'''
def eval_acc(net, device, b_size=10):
    corpus = Corpus('mr', test=True)
    sum_all = 0
    sum_count = 0
    A_E = 1
    for e in range(A_E):
        i = 0
        b = 0
        A_B = int(len(corpus.xs)//b_size)
        
        while i >= 0:
            x, y = batchify(corpus, i, b_size)
            if x is None:
                i = 1
                break
            x, y = x.to(device), y.to(device)
            out = net(x)
            sum_all += (1-(y^torch.argmax(out, dim=-1))).sum().to(torch.device('cpu'))
            sum_count += y.shape[0]
            i += b_size
            b += 1
            # if (b % 50 == 0):
            #     print('E: {}/{} | B: {}/{}'.format(e, A_E, b, A_B))
    return float(sum_all)/float(sum_count)