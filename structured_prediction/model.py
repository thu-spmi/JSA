# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains model structure for structured output prediction experiment.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def repeat(a,n):
    assert len(a.size())==2
    return a.unsqueeze(1).repeat([1,n,1]).view(-1,a.size(1))

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        # encoder
        self.q_layers=nn.Sequential(
            nn.Linear(784,200),nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200,50)
        )
        # learnable prior
        self.prior=nn.Sequential(nn.Linear(784//2,200),nn.Tanh(),
                                 nn.Linear(200, 200), nn.Tanh(),
                                 nn.Linear(200, 50)
                                 )
        # decoder
        self.p_layers=nn.Sequential(
            nn.Linear(50+784//2,200),nn.Tanh(),
            nn.Linear(200, 200), nn.Tanh(),
            nn.Linear(200,784//2)
        )

    def logp_ber(self, x,q_out):
        # q_out is the logit of bernoulli distribution, get log probability of x from the distribution
        prob = - F.binary_cross_entropy_with_logits(q_out,x,reduce=False)
        return torch.sum(prob,1)

    def get_score(self,x,samples):
        # given sample from stochastic layer, denoted as h
        # the condition is the top half, observation is the bottom half, denoted as c,x
        # get logq(h|x,c) and logp(x,h|c)
        cont, ob = x[:, :784 // 2], x[:, 784 // 2:]
        q_out = self.q_layers(x)
        log_q = self.logp_ber(samples, q_out)
        prior_out = self.prior(cont)
        log_prior = self.logp_ber(samples, prior_out)
        z = torch.cat([samples, cont],1)
        out = self.p_layers(z)
        log_p = F.binary_cross_entropy_with_logits(out, ob, reduce=False)
        log_p = log_prior - torch.sum(log_p, 1)
        return log_q, log_p

    def sample_q(self,x):
        # h is sample from stochastic layer
        # the condition is the top half, observation is the bottom half, denoted as c,x
        # get logq(h|x,c), logp(x,h|c) and h (from q(h|x,c))
        cont, ob = x[:,:784//2],x[:,784//2:]
        q_out=self.q_layers(x)
        prob_q=F.sigmoid(q_out)
        samples = (torch.rand_like(prob_q) < prob_q).float()
        log_q=self.logp_ber(samples,q_out)
        prior_out=self.prior(cont)
        log_prior=self.logp_ber(samples,prior_out)
        z=torch.cat([samples,cont],1)
        out=self.p_layers(z)
        log_p=F.binary_cross_entropy_with_logits(out,ob,reduce=False)
        log_p= log_prior - torch.sum(log_p,1)
        return log_q,log_p, samples

    def rws_loss(self,x,n_samples):
        # get loss for RWS
        x=repeat(x,n_samples)
        log_q, log_p,_= self.sample_q(x)
        w=log_p-log_q
        w=(F.softmax(w.view(-1,n_samples),1).view(-1)).detach()
        loss= - torch.mean(w*(log_q+log_p))*n_samples
        return loss
    def vimco_loss(self,x,n_samples):
        # get loss for VIMCO
        x=repeat(x,n_samples)
        log_q, log_p, _ = self.sample_q(x)
        w = (log_p - log_q).view(-1,n_samples)
        w=(w-w.mean(1,keepdim=True)).view(-1).detach()
        loss= - torch.mean(w*log_q)*n_samples/(n_samples-1) - torch.mean(log_p)
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
        # update cache samples by proposal of q(h|x,c)
        log_q, log_p, samples = self.sample_q(x)
        for i in range(len(index)):
            caches[index[i]] = samples[i].cpu()

    def jsa_loss_cache(self,x,n_samples,index,caches):
        # get loss for JSA with cache
        x_single=x
        x = repeat(x, n_samples )

        cache_temp = [caches[i] for i in index]
        cache_samples = torch.stack(cache_temp, 0).cuda()
        log_q, log_p,samples = self.sample_q(x)
        log_q_cache, log_p_cache = self.get_score(x_single, cache_samples)

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
                caches[index[i]]=samples[new_ind].cpu()
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






