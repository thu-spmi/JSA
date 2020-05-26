# Copyright 2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains VIMCO for categorical MNIST experiment.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import argparse
from keras.utils.np_utils import to_categorical 
import warnings
warnings.filterwarnings('ignore')

slim=tf.contrib.slim
Categorical = tf.contrib.distributions.Categorical
Dirichlet = tf.contrib.distributions.Dirichlet
Flatten = tf.keras.layers.Flatten()

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0003, help='lr')
parser.add_argument('--name', default='vimco', help='model name')
parser.add_argument('--nsamples', '-n', type=int, default=20, help='particles number for training')
parser.add_argument('--n_cv', type=int, default=20, help='number of cat var')
parser.add_argument('--n_class', type=int, default=10, help='number of class')
parser.add_argument('--batch', '-b', type=int, default=200, help='mini-batch size')
parser.add_argument('--epoch', '-e', type=int, default=500, help='number of epoch')
parser.add_argument('--seed', type=int, default=0, help='seed')
args = parser.parse_args()
import random
tf.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
#%%
n_for_nll=1000 # particles number for computing NLL
args.name=args.name+'_seed%d'%args.seed
log_variance=True
n = args.nsamples #particles number for training

def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def bernoulli_loglikelihood(b, logits):
    '''
    input: N*d; output: N*d 
    '''
    return b * (-tf.nn.softplus(-logits)) + (1 - b) * (-logits - tf.nn.softplus(-logits))

def categorical_loglikelihood(b, logits):
    '''
    b is N*n_cv*n_class, one-hot vector in row
    logits is N*n_cv*n_class, softmax(logits) is prob
    return: N*n_cv
    '''
    lik_v = b*(logits-tf.reduce_logsumexp(logits,axis=-1,keep_dims=True))
    return tf.reduce_sum(lik_v,axis=-1)
        

def encoder(x,z_dim):
    '''
    return logits [N,n_cv*(n_class-1)]
    z_dim is n_cv*(n_class-1)
    '''
    with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
        h = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        z = tf.layers.dense(h, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim):
    '''
    return logits
    b is [N,n_cv,n_class]
    '''
    with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
        b = Flatten(b)
        h = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        logit_x = tf.layers.dense(h, x_dim, activation = None)
    return logit_x


def fun(x_binary,E,prior_logit0,z_concate):
    '''
    x_binary is N*d_x, E is N*n_cv*n_class, z_concate is N*n_cv*n_class
    prior_logit0 is n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    return (N,)
    '''
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    logits_py = tf.tile(prior_logit1,[tf.shape(E)[0],1,1]) 
    #log p(x|z)
    logit_x = decoder(E,x_dim)
    log_p_x_given_z = tf.reduce_sum(bernoulli_loglikelihood(x_binary, logit_x), axis=1) 
    #log q(z|x)
    log_q_z_given_x = tf.reduce_sum(categorical_loglikelihood(E, z_concate), axis=1)    
    #log p(z)
    log_p_z = tf.reduce_sum(categorical_loglikelihood(E, logits_py), axis=1)
    
    return - log_p_x_given_z - log_p_z + log_q_z_given_x
    

def Fn(pai,prior_logit0,z_concate,x_star_u):
    '''
    pai is [N,n_cv,n_class]
    z_concate is [N,n_class]
    '''
    z_concate1 = tf.expand_dims(z_concate,axis=1)
    E = tf.one_hot(tf.argmin(tf.log(pai+eps)-z_concate1,axis = 3),depth=n_class)
    E = tf.cast(E,tf.float32)
    return fun(x_star_u,E,prior_logit0,z_concate)
     
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(gen_loss0,{x:xs,ntimes:1}))
    return np.mean(cost_eval)

def get_nll(sess,data,total_batch):
    cost_eval = []
    for j in range(total_batch):
        xs, _ = data.next_batch(batch_size)
        cost_eval.append(sess.run(nll_loss, {x: xs,ntimes:n_for_nll}))
    return np.mean(cost_eval)



    
#%% Model
    
tf.reset_default_graph() 

x_dim = 784
eps = 1e-10
n_class = args.n_class ; n_cv = args.n_cv  
z_dim = n_cv * (n_class-1)  
z_concate_dim = n_cv * n_class

prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([n_cv,n_class]))

x = tf.placeholder(tf.float32,[None,x_dim]) 
x_binary = tf.to_float(x > .5)
ntimes = tf.placeholder(tf.int32)
x_binary=tf.tile(x_binary,(ntimes,1))
N = tf.shape(x_binary)[0]

#encoder q(z|x)
z0 = encoder(x_binary,z_dim) 
z = tf.reshape(z0,[N,n_cv,n_class-1])
zeros_logits = tf.zeros(shape = [N,n_cv,1])
z_concate = tf.concat([zeros_logits,z],axis=2)
q_b = Categorical(logits=z_concate)

b_sample = q_b.sample() 
b_sample = tf.one_hot(b_sample,depth=n_class) 
b_sample = tf.cast(b_sample,tf.float32)

#compute decoder p(x|z) gradient 
gen_loss0 = fun(x_binary,b_sample,prior_logit0,z_concate)
nll_loss=tf.reshape(gen_loss0,[n_for_nll,-1])
nll_loss=tf.reduce_logsumexp( - nll_loss,0)-np.log(n_for_nll)
nll_loss= - tf.reduce_mean(nll_loss)
gen_loss = tf.reduce_mean(gen_loss0)
gen_opt = tf.train.AdamOptimizer(args.lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

#compute encoder q(z|x) gradient 

gen_loss1=tf.reshape(-gen_loss0,(n,-1))
gen_loss1=tf.stop_gradient(gen_loss1-tf.reduce_mean(gen_loss1,0))
inf_loss=-tf.reduce_mean(gen_loss0*tf.reshape(gen_loss1,(-1,)))
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')

inf_grads = tf.gradients(inf_loss, inf_vars)
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(args.lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)
if log_variance:
    gen_grads, _ = zip(*gen_gradvars)
    flat_qgrad = tf.concat([tf.reshape(grad, [-1]) for grad in inf_grads if grad is not None], 0)
    flat_pgrad = tf.concat([tf.reshape(grad, [-1]) for grad in list(gen_grads) if grad is not None], 0)

def get_gradient_logvars(sess, xs, samples=1000):
    all_qgrad=[]
    all_pgrad=[]
    for i in range(samples):
        flat_qgrad_temp, flat_pgrad_temp = sess.run([flat_qgrad,flat_pgrad], {x:xs,ntimes:n})
        all_qgrad.append(flat_qgrad_temp)
        all_pgrad.append(flat_pgrad_temp)

    all_qgrad = np.log(np.var(np.stack(all_qgrad,1),1,ddof=1).sum())
    all_pgrad = np.log(np.var(np.stack(all_pgrad, 1), 1, ddof=1).sum())

    return all_qgrad, all_pgrad

with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% data

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
batch_size = args.batch 
training_epochs = args.epoch

#using MNIST-threshold as dataset, i.e. binarization by 0.5 as threshold
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

total_batch = int(mnist.train.num_examples / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)


#%% TRAIN
        
print('Training starts....',args.name)

sess=tf.InteractiveSession()
sess.run(init_op)
import time
begin = time.time()
best_valid=1e4
saver=tf.train.Saver()
from collections import defaultdict
moniter_dict=defaultdict(list)
for epoch in range(training_epochs):
    if log_variance and (epoch%50==0 or epoch==training_epochs-1):
        train_xs, _ = train_data.next_batch(batch_size)
        qlogv, plogv = get_gradient_logvars(sess, train_xs)
        moniter_dict['qlogv'].append(qlogv)
        moniter_dict['plogv'].append(plogv)
        print('log variance:%.1f,%.1f' % (qlogv, plogv))

    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)
        _,cost = sess.run([train_op,gen_loss],{x:train_xs,ntimes:n})
        
    if (epoch+1)%5 == 0:
        valid_nll = get_nll(sess, valid_data, total_valid_batch)
        train_nll = get_nll(sess, train_data, total_batch)
        test_nll = get_nll(sess, test_data, total_test_batch)
        if valid_nll < best_valid:
            best_valid = valid_nll
        moniter_dict['epoch'].append(epoch+1)
        moniter_dict['train_NLL'].append(train_nll)
        moniter_dict['valid_NLL'].append(valid_nll)
        moniter_dict['test_NLL'].append(test_nll)
        print(f'epoch:{epoch+1} time={(time.time()-begin)/60:.3f} train NLL={train_nll:.3f} valid NLL={valid_nll:.3f} test NLL={test_nll:.3f}')
        begin = time.time()

pickle.dump(moniter_dict, open(directory+args.name, 'wb'))
valid_nll,test_nll=min(zip(moniter_dict['valid_NLL'],moniter_dict['test_NLL']))
print("valid test min NLL:%.3f %.3f"%(valid_nll,test_nll))


