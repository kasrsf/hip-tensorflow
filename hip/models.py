import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class TensorHIP():
    """
        Hawkes Intensity Process Model Implemented and Optimized in TensorFlow
        Used for prediction of time series using the Hawkes Self-Excitation 
        model with one or more exogenous sources of influence

        Parameters
        -----------
        x
            a list of the time-series for possible sources of influence in 
            predicting the target series
        y
            target time series
        num_train

        num_test:
    """
    def __init__(self, x, y, num_train, num_test):
        self.x = np.array(x)
        self.y = y
        self.num_train = int(num_train * 0.8)
        self.num_validation = num_train - self.num_train
        self.num_test = num_test
        
        self.gamma = 0
        self.eta = 0
        self.mu = 0
        self.theta = 0
        self.C = 0
                
    def time_decay_base(self, i):
        """
            Kernel Base for the time-decaying exponential kernel
            Increasing per each time step in the series

            Parameters
            ----------
            i
                time series length
        """
        return tf.cast(tf.range(i, 0, -1), tf.float32)

    def get_predictions(self, x_curr, y_hist, eta, mu, theta, C, gamma):
        """
            Predict the future values of X series given the previous values in
            the series and a list of influential series.

            Parameters
            ----------
            x_curr
                 previous values of the series we're trying to predict.
            y_hist
                 a list of the previous values of the relative sources of influence.
            eta, mu, theta, C, gamma
                 model parameters.
        """
        return (
                  eta + tf.reduce_sum(tf.multiply(mu, x_curr))
                  + C * (tf.reduce_sum(y_hist * tf.pow(self.time_decay_base(tf.shape(y_hist)[0]),
                  tf.tile([-1 - theta], [tf.shape(y_hist)[0]]))))
               )       
                
    def fit(self, num_iterations, op='adagrad', verbose=True):
        """
            Fit the best HIP model using multiple random restarts by
            minimizing the loss value of the model 
            
            Parameters
            ----------
            num_iterations
                number of random restarts for fitting the model
            op 
                choice of the optimzier ('adagrad', 'adam')
            verbose 
                print logs

            Returns
            -------
            predictions
                predictions for the time-series
            losses
                loss values for the fitted model
        """
        best_loss = np.inf
        best_eta = 0
        best_mu = 0
        best_theta = 0
        best_C = 0
        
        for i in range(num_iterations):
            if verbose == True:
                print("== Initialization " + str(i + 1))
            losses, e, m, t, c, g = self._fit(op)

            if (losses.sum() < best_loss):
                best_loss = losses.sum()
                best_eta = e
                best_mu = m
                best_theta = t
                best_C = c
                best_gamma = g

        self.eta = best_eta
        self.mu = best_mu
        self.theta = best_theta
        self.C = best_C
        self.gamma = best_gamma
        
        tf.reset_default_graph()
        
        X_CURR = tf.placeholder(tf.float32, name='X_CURR')
        Y_HIST = tf.placeholder(tf.float32, name='Y_HIST')
        Y = tf.placeholder(tf.float32, name='Y')

        eta = tf.get_variable('eta', initializer=tf.constant(best_eta))                        
        mu = tf.get_variable('mu', initializer=tf.constant(best_mu))        
        theta = tf.get_variable('theta', initializer=tf.constant(best_theta))  
        C = tf.get_variable('C', initializer=tf.constant(best_C))
        gamma = tf.get_variable('gamma', initializer=tf.constant(best_gamma))

        pred = self.get_predictions(X_CURR, Y_HIST, eta, mu, theta, C, gamma)
        loss = (tf.square(Y - pred) / 2) + (tf.reduce_sum(tf.square(mu)) / 2)

        if op == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        elif op == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            predictions = np.zeros(len(self.y))
            losses = np.zeros(self.num_test)

            for i in range(self.num_test):
                losses[i] = sess.run(loss, 
                                    feed_dict={
                                        X_CURR: self.x[:, self.num_train + self.num_validation + i], \
                                        Y_HIST: self.y[:self.num_train + self.num_validation + i], \
                                        Y: self.y[self.num_train + self.num_validation + i]
                                        }
                                    )
                
            for i in range(0, len(self.y)):
                predictions[i] = sess.run(pred, 
                                          feed_dict={
                                                    X_CURR: self.x[:, i],
                                                    Y_HIST: self.y[:i], 
                                                    Y: self.y[i]
                                                    }
                                         )
                
            e, m, t, c, g = sess.run([eta, mu, theta, C, gamma])
            
        for i in range(len(predictions)):
            if predictions[i] < 0:
                predictions[i] = 0
                
        self.predictions = predictions
        self.losses = losses

        return predictions, losses
   
    def _fit(self, op='adagrad'):
        """
            Internal method for fitting the model at each iteration of the
            training process
        """
        tf.reset_default_graph()
        
        X_CURR = tf.placeholder(tf.float32, name='X_CURR')
        Y_HIST = tf.placeholder(tf.float32, name='Y_HIST')
        Y = tf.placeholder(tf.float32, name='Y')

        eta = tf.get_variable('eta', shape=(), initializer=tf.random_uniform_initializer(0, 30))                        
        mu = tf.get_variable('mu', shape=(1, len(self.x)), initializer=tf.random_uniform_initializer(-3, 3))        
        theta = tf.get_variable('theta', shape=(), initializer=tf.random_uniform_initializer(0, 30))  
        C = tf.get_variable('C', shape=(), initializer=tf.random_uniform_initializer(0, 30))
        gamma = tf.get_variable('gamma', shape=(), initializer=tf.random_uniform_initializer(0, 30))
        
        pred = self.get_predictions(X_CURR, Y_HIST, eta, mu, theta, C, gamma)

        loss = (tf.square(Y - pred) / 2) + (tf.reduce_sum(tf.square(mu)) / 2)
        prev_loss = None
        iter_counter = 1
        if op == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        elif op == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            while True:  
                for i in range(0, self.num_train):
                    sess.run(optimizer, feed_dict={X_CURR: self.x[:, i], Y_HIST: self.y[:i], Y: self.y[i]})

                losses = np.zeros(self.num_validation)
                for i in range(self.num_validation): 
                    losses[i] = sess.run(loss, feed_dict={X_CURR: self.x[:, self.num_train + i], \
                                                          Y_HIST: self.y[:self.num_train + i], \
                                                          Y: self.y[self.num_train + i]})
                    
                if prev_loss != None and (abs(prev_loss - losses.sum()) < 100000):
                    break
                    
                if iter_counter > 1500:
                    break

                prev_loss = losses.sum()
                iter_counter += 1
                
            e, m, t, c, gamma = sess.run([eta, mu, theta, C, gamma])

        return losses, e, m, t, c, gamma
    
    def plot_predictions(self):
        """
            Plot the current predictions from the fitted model 
        """
        last_index = self.num_train + self.num_test
        
        plt.axvline(self.num_train, color='k')
        plt.plot(np.arange(last_index), self.y[:last_index], 'k--', label='Observed #views')
        if len(self.x) == 1:
            import ipdb; ipdb.set_trace()
            plt.plot(np.arange(last_index), self.x[0][:last_index], 'r--', label='Observed #shares')
        plt.plot(np.arange(last_index), self.predictions[:last_index], 'b-', label='HIP Model Predictions')

        plt.legend()        
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.title("Prediction Vs. Truth")

        plt.show()

    def get_parameters(self):
        """
            Geter method to get the model parameters
        """
        return self.eta, self.mu, self.theta, self.C, self.gamma