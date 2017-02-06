
import warnings

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

warnings.filterwarnings('ignore')

rateLearn = 1e-3
runs = 100
regConst = 1e-5
displaySteps = 1

noiseSigma = 0.01

numSamples = 500
numTest = 250

layers = [2, 20, 20, 20, 2]

def data(n):
    t = np.random.uniform(low = 0.0, high = 2* np.pi, size=n)
    c = np.random.randint(0, high=2, size=n)
    x = t * np.cos(t + c * np.pi) + np.random.normal(0, noiseSigma, n)
    y = t * np.sin(t + c * np.pi) + np.random.normal(0, noiseSigma, n)
    return x, y, c


def defVar(shape, name):
    var = tf.get_variable(name=name,
                          dtype=tf.float32,
                          shape=shape,
                          initializer=tf.random_normal_initializer())
    tf.add_to_collection('modelVars', var)
    tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
    return var


class MultiLayerPercepModel:
    def __init__(self, sess, layers, iterations, learnRate, gamma):
        self.sess = sess
        self.iterations = iterations
        self.learnRate = learnRate
        self.gamma = gamma
        self.layers = layers
        self.buildModel()

    def buildModel(self):
        self.x = tf.placeholder(tf.float32, [None, self.layers[0]])
        self.y = tf.placeholder(tf.float32, [None, self.layers[-1]])

        weights = {}
        biases = {}

        for ii in range(0, len(self.layers) - 1):
            weights[ii] = defVar(name='w%d' % ii, shape=[self.layers[ii], self.layers[ii + 1]])
            biases[ii] = defVar(name='b%d' % ii, shape=[self.layers[ii + 1]])

        self.yhat = tf.nn.relu(tf.add(tf.matmul(self.x, weights[0]), biases[0]))

        for ii in range(1, len(self.layers) - 1):
            self.yhat = tf.nn.relu(tf.add(tf.matmul(self.yhat, weights[ii]), biases[ii]))

        self.yhat = tf.nn.softmax(self.yhat)

        self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yhat, labels=self.y))
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.costs + self.gamma * self.l2_penalty

    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')
        self.optim = (tf.train.AdamOptimizer(learning_rate=self.learnRate).minimize(self.loss, var_list=modelVars))
        self.sess.run(tf.global_variables_initializer())

    def iterateTrainer(self, x, y):
        _, loss = self.sess.run([self.optim, self.loss], feed_dict
        ={self.x: np.transpose(np.asarray(x)), self.y: y})
        return loss

    def train(self, x_samples, y_samples, c_samples):
        for step in range(self.iterations + 1):
            avgCost = 0
            totalBatch = len(x_samples)
            for x, y, c in zip(x_samples, y_samples, c_samples):
                inputs = np.expand_dims(np.asarray([x, y]), axis=1)
                outputs = np.reshape([float(not (c)), float(c)], (1, 2))
                avgCost += self.iterateTrainer(inputs, outputs) / totalBatch
            if step % displaySteps == 0:
                print("Step: {:4d}, Loss: {:.9f}".format(step, avgCost))

    def prob1(self, x1, y1):
        temp = np.expand_dims(np.asarray([x1, y1]), axis=1)
        return self.sess.run(self.yhat, feed_dict={self.x: np.transpose(temp)})[0][1]

    def infer(self, x1, y1):
        return self.prob1(x1, y1) > 0.5


x_samples, y_samples, c_samples = data(numSamples)
x_test, y_test, c_test = data(numTest)

sess = tf.Session()

model = MultiLayerPercepModel(sess=sess, layers=layers, iterations=runs, learnRate=rateLearn, gamma=regConst)
model.initTrainer()
model.train(x_samples, y_samples, c_samples)

total = 0
pred = np.array([])

for x, y, c in zip(x_test, y_test, c_test):
    pred = np.append(pred, model.infer(x, y))

extents = np.arange(-7, 7, 0.1)

xg, yg = np.meshgrid(extents, extents)

f = np.vectorize(model.prob1)
zg = f(xg, yg)

print("Accuracy: {}%".format(np.sum(np.equal(c_test, pred)) * 100 / numTest))
plt.figure()
plt.contour(xg, yg, zg, levels=[0.5])
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.title('Spiral Data Predictions')
plt.plot(x_test[pred == True], y_test[pred == True], 'ro')
plt.plot(x_test[pred == False], y_test[pred == False], 'bo')
plt.show()
