# Krishna Thiyagarajan
# 


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def psi(x, mu, sigma):
	return np.exp(-((x-mu)/sigma)**2)

mu_noise = 0; 
sigma_noise = 0.1; 
N = 50; # Sample size
M = 5; #Hyper variable size

#Create samples
x_data = np.random.uniform(low=0.0, high=1.0, size=N);
noise = np.random.normal(mu_noise, sigma_noise, N);
y_data = np.sin(2*np.pi*x_data) + noise; 

orig_x = np.arange(0.0,1.0, 0.01);
orig_y = np.sin(2*np.pi*orig_x);  
plt.plot(orig_x, orig_y, 'k', x_data, y_data, 'ro')
plt.show(); 


W = tf.Variable(tf.random_uniform([M], -1.0, 1.0)); 
mu = tf.Variable(tf.random_uniform([M], -1.0,1.0)); 
sigma = tf.Variable(tf.random_uniform([M], 0.1, 1.0)); 
b = tf.Variable(tf.zeros[1]); 

yhat = W* psi(x, mu, sigma) + b

# W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# b = tf.Variable(tf.zeros([1]))
# y = W * x_data + b

loss = tf.reduce_mean(0.5 * tf.square(yhat - y_data)); 
optimizer = tf.train.GradientDescentOptimizer(0.1); 
train = optimizer.minimize(loss); 

init = tf.global_variables_initializer(); 
sess = tf.Session(); 
sess.run(init); 

for step in range(10001):
	sess.run(train);
	if step % 100 == 0:
		print step, sess.run(W), sess.run(mu), sess.run(sigma), sess.run(b); 


