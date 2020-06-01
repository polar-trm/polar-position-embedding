from utils import Corpus,  batchify, word_dict
import argparse

from model_loc import *
from model_loc.optmi import ScheduledOptim
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from eval_acc import eval_acc
if torch.cuda.is_available():
    device = 'cuda:3'
else:
    device = 'cpu'
device = torch.device(device)

cof = {"lr":8e-5,
  "attention_probs_dropout_prob": 0.1,
  "rotate_lr": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "type_vocab_size": 2,
  "vocab_size": 18784,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "weight_decay": 0.01,
  "n_warmup_steps": 10000,
  "log_freq": 30
}

MAX_LEN = 500
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', '-b', type=int, default=64)
args = parser.parse_args()

corpus = Corpus('mr')

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.



ntokens = corpus.wd.size
cof['vocab_size'] = ntokens
print("Building BERT model")
bert = BERTPolar(cof['vocab_size'], hidden=cof['hidden_size'], n_layers=cof['num_hidden_layers'], attn_heads=cof['num_attention_heads'], rotate_lr=cof['rotate_lr'])

# Initialize the BERT Language Model, with BERT model
net = BERTPolarLM(bert, cof['vocab_size'])
print("Total Parameters:", sum([p.nelement() for p in net.parameters()]))
# net.load_state_dict(torch.load('./data/out/model/init.pkl'))
net.to(device)
# net = torch.nn.DataParallel(net, device_ids=[0,1,2,3,4,5,6,7])
optim = Adam(net.parameters(), lr=cof['lr'], betas=(cof['adam_beta1'], cof['adam_beta2']), weight_decay=cof['weight_decay'])
optim_schedule = ScheduledOptim(optim, lr=cof['lr'], n_warmup_steps=cof['n_warmup_steps'])

cri = nn.CrossEntropyLoss()
A_E = 500
accs = []
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
        optim_schedule.zero_grad()
        loss = cri(out, y)
        loss.backward()
        optim.step()
        l = optim_schedule.step_and_update_lr(float((A_B*e+b))/float(((A_E)*(A_B))))
        i += args.batch_size
        b += 1
        if (b % 100 == 0):
            print('E: {}/{} | B: {}/{} | Loss: {} | Lr: {} | {}%'.format(e, A_E, b, A_B, loss.item(), l, 100*float((A_B*e+b))/float(((A_E)*(A_B)))))
    if (e % 2 == 0):
        ac = eval_acc(net, device, 10)
        print(ac)
        accs.append(ac)

print(accs)

torch.save(net, './data/out/mr_big.pkl')
