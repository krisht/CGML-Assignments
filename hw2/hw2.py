# coding: utf-8

# # Program for Gaussian Radial Basis Function Regression
# ## Krishna Thiyagarajan
# ## ECE - 411 - Computational Graphs for Machine Learning
# ## Professor Chris Curro
# ## Homework Assignment #1
# ## January 29, 2017

# In[1]:

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import warnings
from IPython.display import Image
from IPython.core.display import HTML

warnings.filterwarnings('ignore')


rateLearn = 1e-2
runs = 50
batchSize = 1000
regConst = 1e-3
displaySteps = 10

layers = [2, 256, 256, 1] # First num is num of inputs, last num is num of outputs


class MultiLayerPercepModel():
    def __init__(self, sess, data, layers, iterations, learnRate, gamma):
        self.sess = sess
        self.data = data
        self.iterations = iterations
        self.learnRate = learnRate
        self.gamma = gamma
        self.layers = layers
        self.buildModel()

    def buildModel(self):

        self.x = tf.placeholder(tf.float32, shape=[None, self.inputs])
        self.y = tf.placeholder(tf.float32, shape =[None, self.classes])

        weights = {}

        biases = {}

        for ii in range(0, len(self.layers)):
            weights['h%d' % ii] = tf.Variable(tf.random_normal([self.layers[ii], self.layers[ii+1]]))

        for ii in range(0, len(self.layers):
            biases['b%d' % ii]  tf.Variable(tf.random_normal([self.layers[ii+1]]))

        yhat = tf.add(tf.matmul(self.x, weights['h0']), biases['b0']);

        for ii in range(1, len(self.layers)):
            yhat = tf.add(tf.matmul(self.x, weights['h%d' % ii]), biases['b%d'%ii]);

        self.mse = tf.reduce_mean('''blah'''));
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'));
        self.loss = self.mse + self.gamma * self.l2_penalty;

    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')
        self.optim = (
        tf.train.GradientDescentOptimizer(learning_rate=self.learnRate).minimize(self.loss, var_list=modelVars))
        self.sess.run(tf.global_variables_initializer())

    def iterateTrainer(self, step, x, y):
        loss, _ = self.sess.run([self.loss, self.optim],
                                feed_dict={self.x: x, self.y: y})
        # if step % 20 == 0:

    # print('Step: {} \t Loss: {}'.format(step, loss))

    def train(self):
        for run in self.iterations:
            avg_cost = 0;
            total_batch = #get the data

            for ii in range(total_batch):
                batch_x, batch_y =
        for step in range(self.iterations + 1):
            for x, y in self.data():
                self.iterateTrainer(step, x, y)

    def infer(self, x):
        y = np.asscalar(self.sess.run(self.yhat, feed_dict={self.x: x}))
        # print(x, y);
        return y;


sess = tf.Session()
model = MultiLayerPercepModel(sess = sess, data = data, iterations=runs, learnRate=rateLearn, gamma=regConst)
model.initTrainer()
model.train()

with tf.variable_scope("", reuse=True):
    w = sess.run(tf.get_variable("w"))
    mu = sess.run(tf.transpose(tf.get_variable("mu")))
    sigma = sess.run(tf.transpose(tf.get_variable("sigma")))
    b = sess.run(tf.get_variable("b"));

print("W =", w);
print("μ =", mu);
print("σ =", sigma);
print("b =", b);

# In[5]:

x_model = np.linspace(0.0, 1.0, 100);

y_model = [];

for a in x_model:
    y_model.append(model.infer(a));
y_model = np.array(y_model);

x_real = np.linspace(0.0, 1.0, 100);
y_real = f(x_real);

examples, targets = zip(*list(data()))

# In[6]:

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 3)
plt.plot(x_real, y_real, 'b-', label='sine')
plt.plot(x_model, y_model, 'r--', label='regression')
plt.plot(np.array(examples), np.array(targets), 'go', label="data");
plt.xlim([0.0, 1.0])
plt.ylim([-1.2, 1.2])
ax.set_xlabel('x')
ax.set_ylabel('y').set_rotation(0)
plt.title('Gaussian RBF Regression of Sine Wave')
plt.tight_layout()
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=3)
plt.show()

# In[7]:

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 3)
ax.set_xlabel('x')
ax.set_ylabel('y').set_rotation(0)
plt.xlim([0.0, 1.0]);
plt.ylim([-2, 2]);
plt.title('Gaussian Basis Curves')
plt.tight_layout()
ax.autoscale(enable=True, axis='y', tight=False)
# Auto scaled to visualize all functions between (0,1)

x_gauss = np.linspace(0.0, 1.0, 100);
for k in range(M):
    with sess.as_default():
        y_gauss = np.asscalar(w[0][k]) * gaussian(x_gauss, mu[0][k], sigma[0][k]).eval();
        lab = "w=%0.3f, mu=%0.3f, sig=%0.3f" % (np.asscalar(w[0][k]), np.asscalar(mu[0][k]), np.asscalar(sigma[0][k]));
    plt.plot(x_gauss, y_gauss, label=lab);
plt.plot(x_gauss, np.full(shape=x_gauss.shape, fill_value=b), label="bias, b=%0.3f" % b);  # Include bias in graph
plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2)
plt.show();

# ## Sample Previous Runs

# In[8]:

Image(url="./regression1.png")

# In[9]:

Image(url="./basis1.png")

# In[10]:

Image(url="./regression2.png")

# In[11]:

Image(url="./basis2.png")


# In[ ]:



