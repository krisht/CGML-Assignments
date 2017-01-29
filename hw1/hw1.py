"""Program for Gaussian RDF Regression

First homework for Computation Graphs for ML
with Professor Chris Curro. An implementation
of Gaussian Radial Basis Function Regression

Variables:
	N {Integer} -- Number of Samples
	M {Integer} -- Number of Basis Functions
	runs {Integer} -- Number of iterations
	rateLearn {Float} -- Learning rate
	regConst {Float} -- Regularization constant
	sigmaNoise {number} -- Std. Dev. of noise on function
	muNoise {number} -- Mean of noise on function
"""

import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#font = {'family' : 'Adobe Caslon Pro',
#        'size'   : 10}

#matplotlib.rc('font', **font)
#

N = 50; # Number of samples
M = 6;  # Hyper parameters
runs = 100; 
rateLearn = 1e-2;
regConst = 0; 
sigmaNoise = 0.1
muNoise = 0

def origFunc(x):
	return np.sin(2 * np.pi * x);

def gaussian(x, mu, sigma):
	return tf.exp(-0.5*(x-mu)**2/sigma**2);

def defVariable(shape, name):
        var = tf.get_variable(name=name,
                                   dtype=tf.float32,
                                   shape=shape,
                                   initializer=tf.random_uniform_initializer(minval=-1.5, maxval=1.5)
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
        
    def buildModel(self):
        self.x = tf.placeholder(tf.float32, shape=[])
        self.y = tf.placeholder(tf.float32, shape=[])
        
        w = defVariable([1, M], 'w')
        mu = defVariable([M,1], 'mu')
        sigma = defVariable([M,1], 'sigma')
        b = defVariable([], 'b')
        phiArr = np.array([])

        for k in range(M):
        	phiArr = np.append(phiArr, gaussian(self.x, mu[k], sigma[k]));
        phi = tf.stack(phiArr.tolist()); 

        self.yhat = b + tf.matmul(w, phi); 
        self.mse = tf.reduce_mean(0.5*tf.square(self.yhat - self.y))
        self.l2_penalty = tf.reduce_sum(tf.get_collection('l2'))
        self.loss = self.mse + self.gamma * self.l2_penalty; 

    def initTrainer(self):
        modelVars = tf.get_collection('modelVars')            
        self.optim = (tf.train.GradientDescentOptimizer(learning_rate=self.learnRate).minimize(self.loss, var_list=modelVars))
        self.sess.run(tf.global_variables_initializer())

    def iterateTrainer(self, step, x, y):
        loss, _ = self.sess.run([self.loss,self.optim],
                                          feed_dict={self.x : x, self.y : y})
        if step % 20 == 0: 
        	print('Step: {} \t Loss: {}'.format(step, loss))

        
    def train(self):
        for step in range(self.iterations+1):
            for x, y in self.data():
                self.iterateTrainer(step, x, y)

    def infer(self, x):
    	y = np.asscalar(self.sess.run(self.yhat, feed_dict={self.x : x}))
    	#print(x, y);
    	return y; 

def data():
    for _ in range(N):
        x = np.random.uniform()
        y  = origFunc(x) + np.random.normal(loc = muNoise, scale = sigmaNoise)
        yield x, y

sess = tf.Session()
model = GaussianRDFModel(sess, data, iterations=runs, learnRate=rateLearn, gamma=regConst)
model.initTrainer()
model.train()

with tf.variable_scope("", reuse = True):
	w = sess.run(tf.get_variable("w"))
	mu = sess.run(tf.transpose(tf.get_variable("mu")))
	sigma = sess.run(tf.transpose(tf.get_variable("sigma")))
	b = sess.run(tf.get_variable("b"));  

print("W =", w); 
print("μ =", mu); 
print("σ =", sigma); 
print("b =", b); 

x_model = np.linspace(0.0, 1.0, 100); 

y_model = []; 

for a in x_model: 
	y_model.append(model.infer(a)); 
y_model = np.array(y_model); 

x_real = np.linspace(0.0, 1.0, 100); 
y_real = origFunc(x_real);  

examples, targets = zip(*list(data()))

fig, ax = plt.subplots(1,1)
fig.set_size_inches(5, 3)
plt.plot(x_real, y_real, '-', x_model, y_model, '-', np.array(examples), np.array(targets), 'o')
plt.xlim([0.0, 1.0])
plt.ylim([-1.2, 1.2])
ax.set_xlabel('x')
ax.set_ylabel('y').set_rotation(0)
plt.title('Gaussian RBF Regression of Sine Wave')
plt.tight_layout()
plt.show(); 


fig, ax = plt.subplots(1,1)
fig.set_size_inches(5, 3)
ax.set_xlabel('x')
ax.set_ylabel('y').set_rotation(0)
plt.title('Individual Gaussian Curves')
plt.tight_layout()

x_gauss = np.linspace(0.0, 1.0, 100); 
for k in range(M):
	with sess.as_default():
		y_gauss = np.asscalar(w[0][k]) * gaussian(x_gauss, mu[0][k], sigma[0][k]).eval() + b;
	plt.plot(x_gauss, y_gauss); 
plt.show(); 

# plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')