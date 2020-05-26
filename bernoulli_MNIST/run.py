# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains main part of Bernoulli MNIST experiment.

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

import argparse
import os
from mnist import MNIST



parser = argparse.ArgumentParser()
parser.add_argument('--sampleNum','-n',type=int,default=1)
parser.add_argument('--batchSize','-b',type=int,default=50)
parser.add_argument('--max_epoch','-e',type=int,default=1000)
parser.add_argument('--layers',type=int,default=2)
parser.add_argument('--method',type=str,default='jsa')
parser.add_argument('--net',type=str,default='nonlinear')
parser.add_argument('--sf',type=str,default='')
parser.add_argument('--seed',type=int,default=0)
args = parser.parse_args()
print(args)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

lr = 3e-4
n_epoch=args.max_epoch
valid_every=5

if args.net=='nonlinear':
    from nonlinear_model import Model
    net=Model(784).cuda()
elif args.net=='linear1':
    from linear_model import Model
    net=Model(784,[200]).cuda()
elif args.net=='linear2':
    from linear_model import Model
    net = Model(784, [200,200]).cuda()

optimizer=torch.optim.Adam(net.parameters(),lr=lr)
# use MNIST-static as dataset, binarization used by (Salakhutdinov & Murray, 2008)
trainset = MNIST(fname="../data/mnist_salakhutdinov.pkl.gz", which_set='train', preproc=[], n_datapoints=50000)
valiset = MNIST(fname="../data/mnist_salakhutdinov.pkl.gz", which_set='valid', preproc=[], n_datapoints=10000)
testset = MNIST(fname="../data/mnist_salakhutdinov.pkl.gz", which_set='test', preproc=[], n_datapoints=10000)

trainx,trainy=torch.FloatTensor(trainset.X),torch.LongTensor(np.arange(0,50000)) #trainy records the index of each training datapoint, for recording the cache samples
validx,validy=torch.FloatTensor(valiset.X),torch.LongTensor(valiset.Y)
testx,testy=torch.FloatTensor(testset.X),torch.LongTensor(testset.Y)
trainset=torch.utils.data.TensorDataset(trainx,trainy)
validset=torch.utils.data.TensorDataset(validx,validy)
testset=torch.utils.data.TensorDataset(testx,testy)

train_loader=DataLoader(trainset,args.batchSize,True,drop_last=True)
train_loader_est=DataLoader(trainset,100)
valid_loader=DataLoader(validset,100)
test_loader=DataLoader(testset,100)

if args.method=='jsa':
    if args.net=='nonlinear':
        caches = torch.FloatTensor(np.random.randint(0, 2, (50000, 200)))
    elif args.net=='linear1':
        caches=[[torch.FloatTensor(np.random.randint(0,2,200)) for _ in range(1)] for _ in range(50000)]
    elif args.net == 'linear2':
        caches = [[torch.FloatTensor(np.random.randint(0, 2, 200)) for _ in range(2)] for _ in range(50000)]

from collections import defaultdict
moniter_dict=defaultdict(list)
start=600 # epoch number to switch from JSA without cache to JSA with cache
def train():
    # training for a epoch
    for x,y in train_loader:
        x=x.cuda()
        if args.method == 'jsa':
            if ep > start:
                loss = net.jsa_loss_cache(x, args.sampleNum, y, caches)
            else:
                loss = net.jsa_loss_nocache(x, args.sampleNum)
        elif args.method == 'rws':
            loss = net.rws_loss(x, args.sampleNum)
        elif args.method=='vimco':
            loss = net.vimco_loss(x, args.sampleNum)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(n=1000,test_loader=None):
    # get NLL for some set
    LL_all=[]
    for x,_ in test_loader:
        x=x.cuda()
        LL=net.get_NLL(x,n)
        LL_all=np.append(LL_all,LL)
    return -np.mean(LL_all)
import time,os
begin=time.time()
os.makedirs('model',exist_ok=True)
os.makedirs('log',exist_ok=True)
os.makedirs('bin',exist_ok=True)
best_nll=1e4

for ep in range(n_epoch):
    train()
    if (ep+1)%valid_every==0:
        valid_nll=test(n=1000,test_loader=valid_loader)
        train_nll=test(n=1000,test_loader=train_loader_est)
        test_nll=test(n=1000,test_loader=test_loader)
        moniter_dict['epoch'].append(ep+1)
        moniter_dict['train_NLL'].append(train_nll)
        moniter_dict['valid_NLL'].append(valid_nll)
        moniter_dict['test_NLL'].append(test_nll)
        print("ep:%d cost:%.3f train valid test NLL:%.3f %.3f %.3f"%(ep+1,time.time()-begin,train_nll,valid_nll,test_nll))
        if valid_nll<best_nll:
            best_nll=valid_nll
        begin=time.time()
valid_nll,test_nll=min(zip(moniter_dict['valid_NLL'],moniter_dict['test_NLL']))
print("valid test min NLL:%.3f %.3f"%(valid_nll,test_nll))

