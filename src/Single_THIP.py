# HIP Implementaion with Single Source of Excitation

import tensorflow as tf
import numpy as np

class Single_TensorHIP():
    def __init__(self, x, y, num_train, num_test):
        self.x = x
        self.y = y
        self.num_train = int(num_train * 0.8)
        self.num_validation = num_train - self.num_train
        self.num_test = num_test
        
        self.eta = 0
        self.mu = 0
        self.theta = 0
        self.C = 0
    
    def time_decay_base(self, i):
        return tf.cast(tf.range(i, 0, -1), tf.float32)

    def get_predictions(self, x_curr, x_hist, eta, mu, theta, C):
        return eta + mu * x_curr + \
            C * (tf.reduce_sum(x_hist * tf.pow(self.time_decay_base(tf.shape(x_hist)[0]), tf.tile([-1 - theta], [tf.shape(x_hist)[0]]))))
        
        
    def fit(self, num_iterations, op='gd'):
        best_loss = np.inf
        best_eta = 0
        best_mu = 0
        best_theta = 0
        best_C = 0
        
        for i in range(num_iterations):
            if i % 5 == 0:
                print("Initialization {0}".format(i + 1))
            losses, e, m, t, c = self._fit(op)

            if (losses.sum() < best_loss):
                best_loss = losses.sum()
                best_eta = e
                best_mu = m
                best_theta = t
                best_C = c
       
        self.eta = best_eta
        self.mu = best_mu
        self.theta = best_theta
        self.C = best_C
        
        tf.reset_default_graph()
        
        X_CURR = tf.placeholder(tf.float32, name='X_CURR')
        X_HIST = tf.placeholder(tf.float32, name='X_HIST')
        Y = tf.placeholder(tf.float32, name='Y')

        eta = tf.get_variable('eta', initializer=tf.constant(best_eta))                        
        mu = tf.get_variable('mu', initializer=tf.constant(best_mu))        
        theta = tf.get_variable('theta', initializer=tf.constant(best_theta))  
        C = tf.get_variable('C', initializer=tf.constant(best_C))

        pred = self.get_predictions(X_CURR, X_HIST, eta, mu, theta, C)

        loss = tf.square(Y - pred, name='loss') / 2

        if op == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss)
        elif op == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        elif op == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            predictions = np.zeros(len(self.x))
            losses = np.zeros(self.num_test)
            for i in range(self.num_test):
                losses[i] = sess.run(loss, feed_dict={X_CURR: self.x[self.num_train + self.num_validation + i], \
                                                      X_HIST: self.x[:self.num_train + self.num_validation + i], \
                                                      Y: self.y[self.num_train + self.num_validation + i]})
                
            for i in range(0, len(self.x)):
                predictions[i] = sess.run(pred, feed_dict={X_CURR: self.x[i], X_HIST: self.x[:i], Y: self.y[i]})
                
            e, m, t, c = sess.run([eta, mu, theta, C])
                
        return predictions, losses
   
    def _fit(self, op='gd'):
        tf.reset_default_graph()
        
        X_CURR = tf.placeholder(tf.float32, name='X_CURR')
        X_HIST = tf.placeholder(tf.float32, name='X_HIST')
        Y = tf.placeholder(tf.float32, name='Y')

        eta = tf.get_variable('eta', shape=(), initializer=tf.random_uniform_initializer(0, 100))                        
        mu = tf.get_variable('mu', shape=(), initializer=tf.random_uniform_initializer(0, 100))        
        theta = tf.get_variable('theta', shape=(), initializer=tf.random_uniform_initializer(2, 50))  
        C = tf.get_variable('C', shape=(), initializer=tf.random_uniform_initializer(0, 50))

        pred = self.get_predictions(X_CURR, X_HIST, eta, mu, theta, C)

        loss = tf.square(Y - pred, name='loss') / 2

        if op == 'gd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(loss)
        elif op == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        elif op == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for i in range(0, self.num_train):
                sess.run(optimizer, feed_dict={X_CURR: self.x[i], X_HIST: self.y[:i], Y: self.y[i]})

            predictions = np.zeros(len(self.x))
            losses = np.zeros(self.num_validation)
            for i in range(self.num_validation):
                losses[i] = sess.run(loss, feed_dict={X_CURR: self.x[self.num_train + i], \
                                                      X_HIST: self.x[:self.num_train + i], \
                                                      Y: self.y[self.num_train + i]})
                
            e, m, t, c = sess.run([eta, mu, theta, C])

        return losses, e, m, t, c
    
    def get_parameters(self):
        return self.eta, self.mu, self.theta, self.C