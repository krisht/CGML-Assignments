
# coding: utf-8

# # Convolutional Neural Network on MNIST Dataset
# ## Krishna Thiyagarajan
# ## ECE - 411 - Computational Graphs for Machine Learning
# ## Professor Chris Curro
# ## Homework Assignment #3a
# ## February 18, 2017

# In[ ]:

import warnings
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
warnings.filterwarnings('ignore')

# Import MNIST data
mnist = input_data.read_data_sets("data/", one_hot=True)

# Hyper Params
runs = 5000

num_inputs = 784
num_classes = 10


# In[ ]:

def def_weight(shape, name):
    var = tf.get_variable(name = name, dtype = tf.float32, shape = shape, initializer = tf.random_normal_initializer())
    tf.add_to_collection('model_vars', var)
    tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
    return var

def def_bias(shape, name):
    var = tf.get_variable(name = name, dtype = tf.float32, shape = shape, initializer = tf.constant_initializer(0.0))
    tf.add_to_collection('model_vars', var)
    tf.add_to_collection('l2', tf.reduce_sum(tf.square(var)))
    return var

weight_dim = {
    'w1': def_weight([5, 5, 1, 32], 'w1'),
    'w2': def_weight([5, 5, 32, 64], 'w2'),
    'w3': def_weight([7 * 7 * 64, 1024], 'w3'),
    'w4': def_weight([1024, num_classes], 'w4')
}

bias_dim = {
    'b1': def_bias([32], 'b1'),
    'b2': def_bias([64], 'b2'),
    'b3': def_bias([1024], 'b3'),
    'b4': def_bias([num_classes], 'b4')
}

# In[ ]:

class MultiLayerConvANNModel:
    def __init__(self, sess, num_in, num_class, weight_dim, bias_dim, iterations, batch_size = 64,  display_steps = 100, learn_rate=1e-3, gamma=1e-4):
        self.sess = sess
        self.num_inputs = num_in
        self.num_classes = num_class
        self.weight_dims = weight_dim
        self.bias_dims = bias_dim
        self.iterations = iterations
        self.batch_size = batch_size
        self.display_steps = display_steps
        self.learn_rate = learn_rate
        self.gamma = gamma
        self.x = tf.placeholder(tf.float32, [None, self.num_inputs])
        self.y = tf.placeholder(tf.float32, [None, self.num_classes])
        self.dropout = tf.placeholder(tf.float32)
        self.build_model()

    def conv2d(self, x, w, b, stride=1):
        x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def build_model(self):
        x = tf.reshape(self.x, shape=[-1, 28, 28, 1])

        self.yhat = self.conv2d(x, self.weight_dims['w1'], self.bias_dims['b1'])
        self.yhat = self.maxpool2d(self.yhat)
        self.yhat = self.conv2d(self.yhat, self.weight_dims['w2'], self.bias_dims['b2'])
        self.yhat = self.maxpool2d(self.yhat)

        self.yhat = tf.reshape(self.yhat, [-1, self.weight_dims['w3'].get_shape().as_list()[0]])
        self.yhat = tf.add(tf.matmul(self.yhat, self.weight_dims['w3']), self.bias_dims['b3'])
        self.yhat = tf.nn.relu(self.yhat)

        self.yhat = tf.nn.dropout(self.yhat, self.dropout)

        self.yhat = tf.add(tf.matmul(self.yhat, self.weight_dims['w4']), self.bias_dims['b4'])

        self.costs = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.yhat, labels=self.y))
        self.l2 = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.costs + self.gamma * self.l2

        self.correct_pred = tf.equal(tf.argmax(self.yhat, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def train(self):
    	model_vars = tf.get_collection('model_vars')
    	self.optim = (tf.train.AdamOptimizer(learning_rate=self.learn_rate).minimize(self.loss, var_list=model_vars))
    	self.sess.run(tf.global_variables_initializer())

    	for kk in range(self.iterations):
    		batch_x, batch_y = mnist.train.next_batch(self.batch_size)
    		self.sess.run([self.optim], feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 0.75})
    		if kk % self.display_steps == 0:
    			loss = self.sess.run(self.loss, feed_dict={self.x: batch_x, self.y: batch_y, self.dropout: 1.0})
    			print("Step: %d, Loss: %f" % (kk, loss))
    	print("Optimization complete!")
    	self.valid_accuracy()

    def valid_accuracy(self):
    	acc = self.sess.run(self.accuracy, feed_dict={self.x: mnist.validation.images[:1000], self.y: mnist.validation.labels[:1000], self.dropout: 1.0})
    	print("Validation Accuracy: ", acc)

    def test_accuracy(self):
    	acc = self.sess.run(self.accuracy, feed_dict={self.x: mnist.test.images[:500], self.y: mnist.test.labels[:500], self.dropout: 1.0})
    	print("Test Accuracy: ", acc)

# In[ ]:

sess = tf.Session()
model = MultiLayerConvANNModel(sess = sess, num_in = num_inputs, num_class = num_classes, weight_dim = weight_dim, bias_dim=bias_dim, iterations = runs, learn_rate=1e-3, gamma=1e-5)
model.train()


# In[ ]:

acc = model.sess.run(model.accuracy, feed_dict={model.x: mnist.test.images[:3000], model.y: mnist.test.labels[:3000], model.dropout: 1.0})
print("Test Accuracy: ", acc)


# In[ ]:



