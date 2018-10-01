import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# stop the optimization process after doing a certain number of iterations
OPTIMIZAITION_MAX_ITERATIONS = 1500
OPTIMIZATION_LOSS_TOLERANCE = 0.005

RANDOM_SEED = 42

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
    def __init__(self, x, y, train_split_size=0.8):
        self.x = np.array(x)
        self.y = np.array(y)

        # do the train-validation-test split 
        # (use same split: Train = 0.8 * 0.8 * length, 
        # validation = 0.2 * 0.8 * length, test = 0.2 * length)
        test_split = int(len(y) * train_split_size)
        validation_split = int(test_split * train_split_size)
        self.train_x, self.train_y = self.x[:, :validation_split], self.y[:validation_split]
        self.validation_x, self.validation_y = (
                                                  self.x[:, validation_split:test_split], 
                                                  self.y[validation_split:test_split]
                                               )  
        self.test_x, self.test_y = self.x[:, test_split:], self.y[test_split:]

        # model parameters
        #self.gamma = 0
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

    def predict(self, x_curr, y_hist, model_params=None,se=None):
        """
            Predict the future values of X series given the previous values in
            the series and a list of influential series.

            Parameters
            ----------
            x_curr
                a list of the previous values of the relative sources of influence.
            y_hist
                previous values of the series we're trying to predict.                 
            mode_params
                 model parameters.
        """
        if model_params is None:
            model_params = self.model_params

        bias = model_params['eta']
        exogenous = tf.reduce_sum(tf.multiply(model_params['mu'], x_curr))
        endogenous = model_params['C'] * tf.reduce_sum(
                                                        y_hist *
                                                        tf.pow(self.time_decay_base(tf.shape(y_hist)[0]), 
                                                               tf.tile([-1 - model_params['theta']], [tf.shape(y_hist)[0]]))
                                                    )
        # TODO: Vectorize Prediction
        return bias + exogenous + endogenous
                
    def train(self, num_iterations, op='adagrad', verbose=True):
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
            best_loss
                best loss value achieved among the iterations
        """
        self.model_params = dict()

        best_loss = np.inf
        for i in range(num_iterations):
            if verbose == True:
                print("== Initialization " + str(i + 1))
            loss_value, model_params = self._fit(iteration_number=i, optimization_algorithm=op)

            if loss_value < best_loss:
                best_loss = loss_value
                self.model_params = model_params

        return best_loss

    def _init_tf_model_variables(self):
        eta = tf.get_variable('eta', initializer=tf.constant(self.model_params['eta']))                        
        mu = tf.get_variable('mu', initializer=tf.constant(self.model_params['mu']))        
        theta = tf.get_variable('theta', initializer=tf.constant(self.model_params['theta']))  
        C = tf.get_variable('C', initializer=tf.constant(self.model_params['C']))
        #gamma = tf.get_variable('gamma', initializer=tf.constant(self.model_params['gamma']))

    def get_predictions(self):
        # predict future values for the test data

        # Instantiate a new model with the trained parameters
        tf.reset_default_graph()
        
        x_observed = tf.placeholder(tf.float32, name='x_observed')
        pred_history = tf.placeholder(tf.float32, name='pred_history')
        y_truth = tf.placeholder(tf.float32, name='y_truth')

        self._init_tf_model_variables()
        
        pred = self.predict(x_observed, pred_history)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            predictions = np.zeros_like(self.y)

            # get model prediction for all of the data
            for index, y in enumerate(self.y):
                predictions[index] = sess.run(
                                                pred, 
                                                feed_dict={
                                                    x_observed: self.x[:, index],
                                                    pred_history: predictions[:index]
                                                    }
                                         )
            
        # TODO: What to do when predictions are zero (enforce max(0, pred)?)
        # for i in range(len(predictions)):
        #     if predictions[i] < 0:
        #         predictions[i] = 0
                
        return predictions
   
    def _fit(self, iteration_number, optimization_algorithm='adagrad'):
        """
            Internal method for fitting the model at each iteration of the
            training process
        """
        tf.reset_default_graph()
        x_observed = tf.placeholder(tf.float32, name='x_observed')
        pred_history = tf.placeholder(tf.float32, name='pred_history')
        y_truth = tf.placeholder(tf.float32, name='y_truth')

        # The model: 
        # eta + sum(mu[i], x_observed[i]) + C * (kernel_base ^ -(1 + theta))
        eta = tf.get_variable(
                              name='eta',
                              shape=(),
                              initializer=tf.random_uniform_initializer(0, 30, seed=RANDOM_SEED + iteration_number)
                            )

        mu = tf.get_variable(
                             name='mu',
                             shape=(1, len(self.x)),
                             initializer=tf.random_uniform_initializer(-3, 3, seed=RANDOM_SEED + iteration_number)
                            )        

        theta = tf.get_variable(
                                name='theta',
                                shape=(),
                                initializer=tf.random_uniform_initializer(-1, 1, seed=RANDOM_SEED + iteration_number)
                               )  

        C = tf.get_variable(
                            name='C',
                            shape=(),
                            initializer=tf.random_uniform_initializer(-1, 1, seed=RANDOM_SEED + iteration_number)
                           )

        # gamma = tf.get_variable(
        #                         name='gamma',
        #                         shape=(),
        #                         initializer=tf.random_uniform_initializer(0, 30, seed=RANDOM_SEED + iteration_number)
        #                        )
        
        # create params dictionary for easier management
        params_keys = ['eta', 'mu', 'theta', 'C']
        params_values = [eta, mu, theta, C]
        params = dict(zip(params_keys, params_values))

        pred = self.predict(x_observed, pred_history, params)
        predictions = np.zeros_like(self.y)
        # TODO: Check effect of adding regularization
        loss = tf.square(y_truth - pred)
        previous_loss = np.inf
        iteration_counter = 1

        if optimization_algorithm == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        elif optimization_algorithm == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=0.5).minimize(loss)
        
        with tf.Session() as sess:
            tf.set_random_seed(RANDOM_SEED)
            sess.run(tf.global_variables_initializer())

            while iteration_counter < OPTIMIZAITION_MAX_ITERATIONS:  
                # First pass, get predictions to be fed for optimization

                # TODO: Do training in one pass. first just create predictions history. in the second run feed the predictions for optimization.
                # Need prediction to work without iteration. To make iterative prediction, create array and conver to tensor. 
                # doesn't seem very efficient. but work for now to see the performance. consult wuga if the model works to improve performance
                # TODO: Vectorize Implementation
                for index, y in enumerate(self.train_y):
                    for i in range(index):
                        predictions[i] = sess.run(
                                                    pred,
                                                    feed_dict={
                                                        x_observed: self.train_x[:, i],
                                                        pred_history: predictions[:i]
                                                    }
                                                )

                    sess.run(
                            optimizer, 
                            feed_dict={
                                    x_observed: self.train_x[:, index],
                                    pred_history: predictions[:index],
                                    y_truth: y
                                }
                            )                            
                
                losses = np.zeros_like(self.validation_y)
                for index, y in enumerate(self.validation_y): 
                    losses[index], predictions[index] = sess.run(
                                            [loss, pred],
                                            feed_dict={
                                                        x_observed: self.validation_x[:, index],
                                                        pred_history: predictions[:index],
                                                        y_truth: y
                                                    }
                                            )
                # Check if optimization iteration produces improvements to the loss value
                # higher than a relative tolerance: tol = |prev_loss - curr_loss| / min(prev_loss, curr_loss)
                curr_loss = losses.sum()
                # TODO: Handle possible division by zero
                relative_loss = abs(previous_loss - curr_loss) / min(previous_loss, curr_loss)
                print(relative_loss, iteration_counter)
                if relative_loss < OPTIMIZATION_LOSS_TOLERANCE: break
                
                previous_loss = losses.sum()
                iteration_counter += 1

            params_vals = sess.run([eta, mu, theta, C])
            fitted_model_params = dict(zip(params_keys, params_vals)) 
        
        return curr_loss, fitted_model_params
    
    def plot_predictions(self):
        """
            Plot the current predictions from the fitted model 
        """
        predictions = self.get_predictions()
        
        data_length = len(self.y)
        data_test_split_point = len(self.train_y) + len(self.validation_y)
        plt.axvline(data_test_split_point, color='k')

        plt.plot(np.arange(data_length), self.y, 'k--', label='Observed #views')

        colors = iter(plt.cm.rainbow(np.linspace(0, 1, self.x.shape[0])))
        for index, exo_source in enumerate(self.x):
            c = next(colors)
            plt.plot(np.arange(data_length), exo_source, c=c, alpha=0.3, label='exo #{0}'.format(index))

        # plot predictions on training data with a different alpha to make the plot more clear            
        plt.plot(
                    np.arange(data_test_split_point),
                    predictions[:data_test_split_point], 
                    'b-',
                    alpha=0.3,
                    label='Model Fit'
                )
        plt.plot(
                    np.arange(data_test_split_point, data_length),
                    predictions[data_test_split_point:], 
                    'b-',
                    alpha=1,
                    label='Model Predictions'
                )


        plt.legend()        
        plt.xlabel('Time')
        plt.ylabel('Y')
        plt.title("Prediction Vs. Truth")

        plt.show()

    def get_model_parameters(self):
        """
            Getter method to get the model parameters
        """
        return self.model_params