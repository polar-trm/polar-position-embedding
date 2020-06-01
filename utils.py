import pandas as pd
import os
import pickle
import torch
import numpy as np
def load(dataset='mr', filter=False):
    data_dir = "data/" + dataset
    datas = []
    for data_name in ['train.csv', 'dev.csv', 'test.csv']:
        if data_name == 'train.csv':
            data_file = os.path.join(data_dir, data_name)
            data = pd.read_csv(data_file, header=None, sep="\t", names=[
                               "question", "flag"], quoting=3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)
        if data_name == 'dev.csv':
            data_file = os.path.join(data_dir, data_name)
            data = pd.read_csv(data_file, header=None, sep="\t", names=[
                               "question", "flag"], quoting=3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)
        if data_name == 'test.csv':
            data_file = os.path.join(data_dir, data_name)
            data = pd.read_csv(data_file, header=None, sep="\t", names=[
                               "question", "flag"], quoting=3).fillna("WASHINGTON")
            if filter == True:
                datas.append(removeUnanswerdQuestion(data))
            else:
                datas.append(data)

    sub_file = os.path.join(data_dir, 'submit.txt')
    return tuple(datas)

def removeUnanswerdQuestion(df):
    counter = df.groupby("question").apply(lambda group: sum(group["flag"]))
    questions_have_correct = counter[counter > 0].index
    counter = df.groupby("question").apply(
        lambda group: sum(group["flag"] == 0))
    questions_have_uncorrect = counter[counter > 0].index
    counter = df.groupby("question").apply(lambda group: len(group["flag"]))
    questions_multi = counter[counter > 1].index

    return df[df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_correct) & df["question"].isin(questions_have_uncorrect)].reset_index()

class word_dict():
    def __init__(self):
        self.i2w = ['_pad_', '_cls_']
        self.w2i = {'_pad_': 0, '_cls_': 1}
        self.size = 2
    def add(self, w):
        if w not in self.i2w:
            self.w2i[w] = self.size
            self.size += 1
            self.i2w.append(w)


def build_dict(dataname='mr', clean=False, dic_name='voc.dict'):
    train, dev, test = load(dataset=dataname, filter=clean)
    all_token = ' '.join([' '.join(train['question']), ' '.join(test['question'])])
    wd = word_dict()
    all_token = all_token.replace('\n', ' ').replace('.', ' ').replace('  ', ' ').split(' ')
    for w in all_token:
        wd.add(w)
    with open('data/'+dataname+'/'+dic_name, 'wb+') as f:
        pickle.dump(wd, f)
def get_len(dataname='mr', clean=False, dic_name='voc.dict'):
    train, dev, test = load(dataset=dataname, filter=clean)
    train_len=[]
    for i in range(len(train)):
        x = train.iloc[i,0].replace('\n', ' ').replace(',', ' ').replace('.', ' ').replace('  ', ' ').split(' ')
        train_len.append(len(x))
    for i in range(len(test)):
        x = test.iloc[i,0].replace('\n', ' ').replace(',', ' ').replace('.', ' ').replace('  ', ' ').split(' ')
        train_len.append(len(x))
    print('train len: {}, {}, mean: {}'.format(min(train_len), max(train_len), np.sum(train_len)/(len(test)+len(train))))
    print(len(train))


class Corpus():
    def __init__(self, dataname='mr', clean=False, max_len=64, test=False):
        if not test:
            train_t, _, test_t = load(dataset=dataname, filter=clean)
        else:
            test_t, _, train_t = load(dataset=dataname, filter=clean)

        with open('data/'+dataname+'/voc.dict', 'rb') as f:
            self.wd = pickle.load(f)

        self.xs = []
        self.ls = []
        for i in range(len(train_t)):
            x = train_t.iloc[i,0].replace('\n', ' ').replace(',', ' ').replace('.', ' ').replace('  ', ' ').split(' ')
            x = ['_cls_'] + x
            x = self.trans2id(x)
            x = self.padding(x, max_len)
            y = train_t.iloc[i,1]
            self.xs.append(x)
            self.ls.append(y)
    def trans2id(self, x):
        ids = []
        for i in x:
            i = i.strip()
            if (len(i)<=0):
                i = '_pad_'
            ids.append(self.wd.w2i[i])
        return ids

    def padding(self, x, l):
        if len(x) > l:
            x = x[:l]
        elif len(x) < l:
            x = x + [0]*(l-len(x))
        return x

def batchify(c, s, b):
    if s >= len(c.ls):
        return None, None
    if s + b > len(c.ls):
        b = len(c.ls) - s
    x = c.xs[s:s+b]
    y = c.ls[s:s+b]
    return torch.tensor(x), torch.tensor(y)

if __name__ == "__main__":
    build_dict('mr')
