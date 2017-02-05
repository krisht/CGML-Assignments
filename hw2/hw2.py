

import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

warnings.filterwarnings('ignore')

rateLearn = 1e-2
runs=50
batchSize=1000
regConst=1e-3
displaySteps=10

totalSamples=1000
trainSamples=700

layers = [2, 4, 8, 16, 32, 32, 16, 8, 4, 2]

def dataTrain():
    for _ in range(trainSamples):
        t=np.random.uniform(low=0, high=4*np.pi)
        p=np.random.randint(0, high=2)
        x1=t*np.cos(t+p*np.pi) + np.random.normal(scale=0.01)
        x2=t*np.sin(t+p*np.pi) + np.random.normal(scale=0.01)
        y =p
        ynot = 1 - p
        yield x1, x2, y, ynot

def defWeight(shape, name):
    var = tf.get_variable(name=name,
                          dtype=tf.float32,
                          shape=shape,
                          initializer=tf.random_uniform_initializer(minval=-1, maxval=1)
                          # Works better as U(-1,1) as oppoed to N(0, 0.1)
                          )
    tf.add_to_collection('modelVars', var)
    tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
    return var

def defBias(shape, name):
    var = tf.get_variable(name=name,
                          dtype=tf.float32,
                          shape=shape,
                          initializer=tf.random_normal_initializer(stddev=0.1)
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
        self.x = tf.placeholder(tf.float32, shape = [None, self.layers[0]])
        self.y = tf.placeholder(tf.float32, shape = [None, self.layers[len(self.layers)-1]])

        weights = {}
        biases = {}

        for ii in range(0, len(self.layers)-1):
            weights[ii] = defWeight(name='w%d' % ii, shape = [self.layers[ii], self.layers[ii+1]])

        for ii in range(0, len(self.layers)-1):
            biases[ii] = defBias(name='b%d' % ii, shape = [self.layers[ii+1]])
        self.yhat = tf.nn.relu(tf.add(tf.matmul(self.x, weights[0]), biases[0]))

        for ii in range(1, len(self.layers)-1):
            self.yhat = tf.nn.relu(tf.add(tf.matmul(self.yhat, weights[ii]), biases[ii]));

        self.yhat = tf.sigmoid(self.yhat);

        self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.yhat, labels = self.y));
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.costs + self.gamma * self.l2_penalty

    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')
        self.optim = (tf.train.GradientDescentOptimizer(learning_rate=self.learnRate).minimize(self.loss, var_list = modelVars))

        self.sess.run(tf.global_variables_initializer())

    def iterateTrain(self, x, y):

        loss = self.sess.run(self.loss, feed_dict={self.x: np.transpose(np.asarray(x)), self.y: np.transpose(y)})
        print("Loss: {}".format(loss))

    def train(self):
        for _ in range(self.iterations):
            for x1, x2, y, ynot in self.data:
                self.iterateTrain([[x1],[x2]], [[y],[ynot]])

    def infer(self, x):
        y = np.asscalar(self.sess.run(self.yhat, feed_dict={self.x: x}))
        return y; 

sess = tf.Session()
model = MultiLayerPercepModel(sess = sess, data = dataTrain(), iterations=runs, learnRate=rateLearn, gamma=regConst, layers=layers)
model.initTrainer()
model.train()

print(tf.get_collection('modelVars'))