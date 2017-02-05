# coding: utf-8

# # Program for Gaussian Radial Basis Function Regression
# ## Krishna Thiyagarajan
# ## ECE - 411 - Computational Graphs for Machine Learning
# ## Professor Chris Curro
# ## Homework Assignment #1
# ## January 29, 2017

# In[1]:

import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image

warnings.filterwarnings('ignore')


rateLearn = 1e-2
runs = 50
batchSize = 1000
regConst = 1e-3
displaySteps = 10

totalSamples = 1000
trainSamples = 700

layers = [2, 256, 256, 1] # First num is num of inputs, last num is num of outputs


def dataTrain():
    for _ in range(trainSamples):
        t=np.random.uniform(low = 0, high=4*np.pi);
        p=np.random.randint(0,high=2);
        x1=t*np.cos(t+p*np.pi) + np.random.normal(scale=0.01);
        x2=t*np.sin(t+p*np.pi)+ np.random.normal(scale=0.01);
        y = p;
        yield x1,x2,y;

def dataTest():
    for _ in range(totalSamples - trainSamples):
        t=np.random.uniform(low = 0, high=4*np.pi);
        p=np.random.randint(0,high=2);
        x1=t*np.cos(t+p*np.pi) + np.random.normal(scale=0.01);
        x2=t*np.sin(t+p*np.pi)+ np.random.normal(scale=0.01);
        y = p;
        yield x1,x2,y;


def defVariable(shape, name):
    var = tf.get_variable(name=name,
                          dtype=tf.float32,
                          shape=shape,
                          initializer=tf.random_uniform_initializer(minval=-1, maxval=1)
                          # Works better as U(-1,1) as oppoed to N(0, 0.1)
                          )
    tf.add_to_collection('modelVars', var)
    tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
    return var

class MultiLayerPercepModel:
    def __init__(self, sess, data, layers, iterations, learnRate, gamma):
        self.sess = sess
        self.data = data
        self.iterations = iterations
        self.learnRate = learnRate
        self.gamma = gamma
        self.layers = layers
        self.buildModel()

    def buildModel(self):

        self.x = tf.placeholder(tf.float32, shape=[None, self.layers[0]])
        self.y = tf.placeholder(tf.float32, shape =[None, self.layers[len(self.layers)-1]])

        weights = {}

        biases = {}

        for ii in range(0, len(self.layers)):
            weights[ii] = defVariable(name = 'w%d' % ii, shape = [self.layers[ii], self.layers[ii+1]])

        for ii in range(0, len(self.layers)):
            biases[ii] = defVariable(name = 'b%d' % ii, shape = [self.layers[ii+1]]);

        self.yhat = tf.nn.relu(tf.add(tf.matmul(self.x, weights['h0']), biases['b0']))

        for ii in range(1, len(self.layers)-1):
            self.yhat = tf.relu(tf.add(tf.matmul(self.yhat, weights['h%d' % ii]), biases['b%d'%ii]));

        self.yhat = tf.sigmoid(tf.add(tf.matmul(self.yhat, weights['h%d'% (len(self.layers)-1)]), biases['b%d' % (len(
            self.layers)-1)]));

        self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.yhat, labels = self.y));
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.costs + self.gamma * self.l2_penalty

    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')
        self.optim = (tf.train.GradientDescentOptimizer(learning_rate=self.learnRate).minimize(self.loss, var_list =
        modelVars))

        self.sess.run(tf.global_variables_initializer())

    def iterateTrain(self, x1, x2, y):
        loss = self.sess.run(self.loss, feed_dict={self.x : [x1, x2], self.y=y})

        print("Loss: {}".format(loss)); 

    def train(self):
        for _ in range(self.iterations):
            for x1, x2, y, in self.data():
                self.train_iter(x1,x2,y);

    def infer(self, x):
        y = np.asscalar(self.sess.run(self.yhat, feed_dict={self.x: x}))
        # print(x, y);
        return y;


# sess = tf.Session()
# model = MultiLayerPercepModel(sess = sess, data = data(), iterations=runs, learnRate=rateLearn, gamma=regConst)
# model.initTrainer()
# model.train()



x1,x2,y = zip(*dataTrain());

fig, ax = plt.subplots(1,1)
fig.set_size_inches(5,3)
plt.plot(x1,x2, 'o')
plt.xlim([-2.1,2.1])
plt.ylim([-2.1,2.1])
ax.set_xlabel('x');
ax.set_ylabel('y').set_rotation(0)
plt.title('Curves');
plt.tight_layout();
plt.show()