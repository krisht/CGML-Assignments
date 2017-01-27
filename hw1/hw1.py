#!/bin/python

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import sys


#font = {'family' : 'Adobe Caslon Pro',
#        'size'   : 10}

#matplotlib.rc('font', **font)
#

N = 50; # Number of samples
M = 5;  # Hyper parameters



def defVariable(shape, name):
        var = tf.get_variable(name=name,
                                   dtype=tf.float32,
                                   shape=shape,
                                   initializer=tf.random_normal_initializer(stddev=0.1)
        )
        tf.add_to_collection('modelVars', var)
        tf.add_to_collection('l2', tf.reduce_sum(tf.pow(var,2)))
        return var
    
class GaussianRDFModel():
    def __init__(self, sess, data, iterations, learnRate, gamma):
        self.sess = sess
        self.data = data
        self.iterations = iterations
        self.learnRate = learnRate
        self.gamma = gamma
        self.buildModel()

    def gaussian(self, x, mu, sigma): 
    	return tf.exp(-(x-mu)**2/sigma**2);
        
    def buildModel(self):
        self.x = tf.placeholder(tf.float32, shape=[])
        self.y = tf.placeholder(tf.float32, shape=[])
        
        w = defVariable([1, M], 'w')
        mu = defVariable([M,1], 'mu')
        sigma = defVariable([M,1], 'sigma')
        b = defVariable([], 'b')
        phiArr = np.array([])

        for k in range(M):
        	phiArr = np.append(phiArr, self.gaussian(self.x, mu[k], sigma[k]));
        phi = tf.stack(phiArr.tolist()); 

        self.yhat = b + tf.matmul(w, phi); 

        self.mse = tf.reduce_mean(tf.square(self.yhat - self.y))
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.mse + self.gamma*self.l2_penalty
        
    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')            
        self.optim = (tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss, var_list=modelVars))
        self.sess.run(tf.global_variables_initializer())

    def iterateTrainer(self, step, x, y):
        loss, _ = self.sess.run([self.loss,self.optim],
                                          feed_dict={self.x : x, self.y : y})
        if step % 5 == 0: 
        	print('Step: {} \t Loss: {}'.format(step, loss))

        
    def train(self):
        for step in range(self.iterations):
            for x, y in self.data():
                self.iterateTrainer(step, x, y)

    def infer(self, x):
        return self.sess.run(self.yhat, feed_dict={self.x : x})

def data():
    sigmaNoise = 0.5
    for _ in range(N):
        x = np.random.uniform()
        y  = np.sin(2 * np.pi * x) + np.random.normal(loc = 0, scale = sigmaNoise)
        yield x, y

sess = tf.Session()
model = GaussianRDFModel(sess, data, iterations=101, learnRate=1e-2, gamma=1e-4)
model.initTrainer()
model.train()
    
with tf.variable_scope("", reuse = True):
	print("W =", sess.run(tf.get_variable("w"))); 
	print("Mu =", sess.run(tf.transpose(tf.get_variable("mu"))))
	print("Sigma =", sess.run(tf.transpose(tf.get_variable("sigma"))))
	print("b =", sess.run(tf.get_variable("b")))