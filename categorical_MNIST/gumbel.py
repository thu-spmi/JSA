# Copyright 2019 ARM-gradient (https://github.com/ARM-gradient/ARSM)
#           2020 Tsinghua University, Author: Yunfu Song
# Apache 2.0.
# This script contrains ST-Gumbel-Softmax for categorical MNIST experiment.

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.0003, help='lr')
parser.add_argument('--nsamples', '-n', type=int, default = 20, help='particles number for training')
parser.add_argument('--name', '-n', default='stgs', help='model name')
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
slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical

#%%
directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)     
EXPERIMENT=args.name+'_seed%d'%args.seed

batch_size = 200
training_epochs = args.epoch
lr = args.lr
x_dim = 784
z_dim = 200
K = 10     #number of cat variable
C = z_dim//K  #C-way
learn_temp = True
n_for_nll=1000 # particles number for computing NLL
log_variance=True
n = args.nsamples #particles number for training

def lrelu(x, alpha=0.2):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def encoder(x,z_dim):
    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):
        h = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h, z_dim, activation=None)
    return log_alpha

def decoder(b,x_dim):
    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
        h = slim.stack(b ,slim.fully_connected,[256,512],activation_fn=lrelu)
        log_alpha = tf.layers.dense(h, x_dim, activation=None)
    return log_alpha
def get_nll(sess,data,total_batch):
    cost_eval = []
    for j in range(total_batch):
        xs, _ = data.next_batch(batch_size)
        cost_eval.append(sess.run(nll, {x0: xs,ntimes:n_for_nll}))
    return np.mean(cost_eval)
#%%
tf.reset_default_graph() 

x0 = tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
x = tf.to_float(x0 > .5)
ntimes = tf.placeholder(tf.int32)
x=tf.tile(x,(ntimes,1))
logits_y = tf.reshape(encoder(x,z_dim),[-1,C,K])
tau = tf.Variable(1.0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()

y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
y = tf.stop_gradient(y_hard - y) + y
net = slim.flatten(y)

logits_x = decoder(net,x_dim)

p_x = Bernoulli(logits=logits_x)
x_mean = p_x.mean()

recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K #uniform

p_cat_y = OneHotCategorical(logits=logits_py)
q_cat_y = OneHotCategorical(logits=logits_y)
KL_qp =  tf.distributions.kl_divergence(q_cat_y, p_cat_y)

KL = tf.reduce_sum(KL_qp,1)

neg_elbo0 = KL - recons
neg_elbo = neg_elbo0[:,np.newaxis]
loss = tf.reduce_mean(KL - recons)
nll= tf.reshape( - neg_elbo0,[n_for_nll,-1])
nll=tf.reduce_logsumexp(nll,0)-np.log(n_for_nll)
nll= - tf.reduce_mean(nll)
gs_grad = tf.gradients(loss, logits_y)
train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init_op=tf.global_variables_initializer()
if log_variance:
    inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
    q_grads = tf.gradients(loss, inf_vars)
    p_grad=tf.gradients(loss, gen_vars)
    flat_qgrad= tf.concat([tf.reshape(grad, [-1]) for grad in q_grads if grad is not None], 0)
    flat_pgrad=tf.concat([tf.reshape(grad, [-1]) for grad in p_grad if grad is not None], 0)

#%% TRAIN
#using MNIST-threshold as dataset, i.e. binarization by 0.5 as threshold
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)

def get_gradient_logvars(sess, xs, samples=1000):
    all_qgrad=[]
    all_pgrad=[]
    for i in range(samples):
        flat_qgrad_temp, flat_pgrad_temp = sess.run([flat_qgrad,flat_pgrad], {x: xs, ntimes: n})
        all_qgrad.append(flat_qgrad_temp)
        all_pgrad.append(flat_pgrad_temp)

    all_qgrad = np.log(np.var(np.stack(all_qgrad,1),1,ddof=1).sum())
    all_pgrad = np.log(np.var(np.stack(all_pgrad, 1), 1, ddof=1).sum())

    return all_qgrad, all_pgrad

def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs, _ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(neg_elbo0,{x0:xs,ntimes:1}))
    return np.mean(cost_eval)

print('Training starts....',EXPERIMENT)

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
        _,cost,_ = sess.run([train_op,loss,tau],{x0:train_xs,ntimes:n})
        
    if (epoch+1)%5 == 0:
        valid_nll = get_nll(sess, valid_data, total_valid_batch)
        train_nll = get_nll(sess, train_data, total_batch)
        test_nll = get_nll(sess, test_data, total_test_batch)
        valid_nll, train_nll, test_nll = 0, 0, 0
        if valid_nll < best_valid:
            best_valid = valid_nll
        moniter_dict['epoch'].append(epoch + 1)
        moniter_dict['train_NLL'].append(train_nll)
        moniter_dict['valid_NLL'].append(valid_nll)
        moniter_dict['test_NLL'].append(test_nll)
        print(f'epoch:{epoch+1} time={(time.time()-begin)/60:.3f} train NLL={train_nll:.3f} valid NLL={valid_nll:.3f} test NLL={test_nll:.3f} ')
        begin = time.time()

pickle.dump(moniter_dict, open(directory+EXPERIMENT, 'wb'))
valid_nll,test_nll=min(zip(moniter_dict['valid_NLL'],moniter_dict['test_NLL']))
print("valid test min NLL:%.3f %.3f"%(valid_nll,test_nll))
