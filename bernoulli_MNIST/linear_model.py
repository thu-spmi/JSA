# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains linear model structure for Bernoulli MNIST experiment.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def repeat(a,n):
    assert len(a.size())==2
    return a.unsqueeze(1).repeat([1,n,1]).view(-1,a.size(1))

class Model(nn.Module):
    def __init__(self,n_vis,layers):
        self.layers=layers
        super(Model,self).__init__()
        self.q_layers=[nn.Linear(n_vis,layers[0])]  #encoder
        for i in range(len(layers)-1):
            self.q_layers.append(nn.Linear(layers[i],layers[i+1]))
        self.p_layers=[]  # decoder
        for i in range(len(layers)-1,0,-1):
            self.p_layers.append(nn.Linear(layers[i],layers[i-1]))
        self.p_layers.append(nn.Linear(layers[0],n_vis))
        print(self.p_layers[0].weight.data.numpy()[:5,:5])
        for layer in self.q_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias,-1.0)
        for layer in self.p_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias,-1.0)
        print(self.p_layers[0].weight.data.numpy()[:5, :5])
        self.q_layers=nn.ModuleList(self.q_layers)
        self.p_layer=nn.ModuleList(self.p_layers)
    def forward(self,x):
        return self.sample_q(x)
    def sample(self,layer,x):
        # input x through layer, get a sample from the output distribution and the log probability
        prob=layer(x)
        u=torch.rand_like(prob)
        y=(F.sigmoid(prob)>u).float()
        log_prob = - F.binary_cross_entropy_with_logits(prob,y,reduce=False)
        return y,log_prob.sum(1)
    def log_prob(self,layer,x,y):
        # input x through layer, get log probability of sample y from the output distribution
        prob = layer(x)
        log_prob = - F.binary_cross_entropy_with_logits(prob, y, reduce=False)
        return log_prob.sum(1)
    def bernolli_prior(self,x):
        #get prior log log probability
        prob=torch.ones_like(x)
        log_prob = prob * np.log(0.5)
        return log_prob.sum(1)
    def get_score(self,x,y_lists):
        # y_lists is the samples from all stochastic layers, denoted as h, get logq(h|x) and logp(x,h)
        log_q_all = self.log_prob(self.q_layers[0],x,y_lists[0])
        for i in range(len(self.q_layers)-1):
            log_q_all += self.log_prob(self.q_layers[i+1],y_lists[i],y_lists[i+1])
        log_p_all = self.bernolli_prior(y_lists[-1])
        for i in range(len(self.p_layers)-1):
            log_p_all += self.log_prob(self.p_layers[i], y_lists[len(self.p_layers) - i - 1], y_lists[len(self.p_layers) - i - 2])
        log_p_all+=self.log_prob(self.p_layers[-1], y_lists[0], x)
        return log_q_all, log_p_all
    def sample_q(self,x):
        # h is the samples from all stochastic layers, get logq(h|x), logp(x,h) and h (from q(h|x))
        samples=[0]*(len(self.q_layers)+1)
        log_q_all=0
        samples[0]=x
        for i in range(len(self.q_layers)):
            samples[i+1],log_q = self.sample(self.q_layers[i],samples[i])
            log_q_all+= log_q
        log_p_all = self.bernolli_prior(samples[-1])
        for i in range(len(self.p_layers)):
            log_p_all += self.log_prob(self.p_layers[i],samples[len(self.p_layers)-i],samples[len(self.p_layers)-i-1])
        return log_q_all,log_p_all, samples[1:]
    def vimco_loss(self,x,n_samples):
        # get loss for VIMCO
        x=repeat(x,n_samples)
        log_q, log_p, _ = self.sample_q(x)
        w = (log_p - log_q).view(-1,n_samples)
        w=(w-w.mean(1,keepdim=True)).view(-1).detach()
        loss= - torch.mean(w*log_q)*n_samples/(n_samples-1) - torch.mean(log_p)
        return loss
    def rws_loss(self,x,n_samples):
        #get loss for RWS
        x=repeat(x,n_samples)
        log_q, log_p,_= self.sample_q(x)
        w=log_p-log_q
        w=(F.softmax(w.view(-1,n_samples),1).view(-1)).detach()
        loss= - torch.mean(w*(log_q+log_p))*n_samples
        return loss
    def jsa_loss_nocache(self,x,n_samples):
        # get loss for JSA without cache
        x=repeat(x,n_samples)
        log_q, log_p,_= self.sample_q(x)
        logw = (log_p - log_q).detach().cpu().numpy()
        ind=np.arange(0,len(logw)).reshape(-1,n_samples)
        logw=logw.reshape(-1,n_samples)
        w_cur=logw[:,0]
        ind_cur=ind[:,0]
        ind_all=[]
        for i in range(n_samples-1):
            u=np.random.uniform(size=len(ind))
            cond=u<np.exp(logw[:,i+1]-w_cur)
            ind_cur=ind_cur*(1-cond)+ind[:,i+1]*cond
            w_cur = w_cur * (1 - cond) + logw[:, i + 1] * cond
            ind_all=np.append(ind_all,ind_cur)
        ind=torch.LongTensor(ind_all).cuda()
        log_q = torch.gather(log_q, dim=0, index=ind)
        log_p = torch.gather(log_p, dim=0, index=ind)
        loss = - torch.mean(log_q) - torch.mean(log_p)
        return loss

    def update_cache(self,x,index,caches):
        # update cache samples by proposal of q(h|x)
        log_q, log_p, samples = self.sample_q(x)
        for i in range(len(index)):
            caches[index[i]] = [samples[k][i].cpu() for k in range(len(self.layers))]

    def jsa_loss_cache(self,x,n_samples,index,caches):
        # get loss for JSA with cache
        x_single=x
        x = repeat(x, n_samples )
        cache_lists=[]
        for j in range(len(self.layers)):
            cache_temp=[caches[i][j] for i in index]
            cache_lists.append(torch.stack(cache_temp,0).cuda())
        log_q, log_p,samples = self.sample_q(x)
        log_q_cache, log_p_cache = self.get_score(x_single, cache_lists)

        log_q = log_q.view(-1, n_samples )
        log_q_cache=log_q_cache.view(-1,1)
        log_q=torch.cat([log_q_cache,log_q],1).view(-1)

        log_p = log_p.view(-1, n_samples )
        log_p_cache = log_p_cache.view(-1, 1)
        log_p = torch.cat([log_p_cache, log_p], 1).view(-1)

        logw = (log_p - log_q).detach().cpu().numpy()

        ind = np.arange(0, len(logw)).reshape(-1, n_samples +1)
        logw = logw.reshape(-1, n_samples +1)
        w_cur = logw[:,0]
        ind_cur = ind[:, 0]
        ind_all=[]
        for i in range(n_samples):
            u = np.random.uniform(size=len(ind))
            cond = u < np.exp(logw[:, i + 1] - w_cur)
            ind_cur = ind_cur * (1 - cond) + ind[:, i + 1] * cond
            w_cur = w_cur * (1 - cond) + logw[:, i + 1] * cond
            ind_all = np.append(ind_all, ind_cur)
        for i in range(len(index)):
            ind_temp=int(ind_cur[i])
            mod=ind_temp%(n_samples+1)
            if mod!=0:
                new_ind= ind_temp//(n_samples+1)*(n_samples)+mod-1
                caches[index[i]]=[samples[k][new_ind].cpu() for k in range(len(self.layers))]
        ind = torch.LongTensor(ind_all).cuda()
        log_q = torch.gather(log_q, dim=0, index=ind)
        log_p = torch.gather(log_p, dim=0, index=ind)
        loss = - torch.mean(log_q) - torch.mean(log_p)
        return loss

    def get_NLL(self,x,n_samples=100):
        # computing NLL
        x = repeat(x, n_samples)
        log_q, log_p, _ = self.sample_q(x)
        w = log_p - log_q
        w=w.view(-1,n_samples)
        w=torch.logsumexp(w,1)-np.log(n_samples)
        return w.detach().cpu().numpy()






