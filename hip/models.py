import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# stop the optimization process after doing a certain number of iterations
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-6, 1e-6, 1e-6
PRINT_ITERATION_CHECKPOINT_STEPS = 100

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
    def __init__(self, 
                 xs,
                 ys=None,
                 train_split_size=0.8,
                 l2_param=0,
                 learning_rate=0.1,
                 num_initializations=5,
                 max_iterations=1000,
                 params=None,
                 verbose=True
                ):
        self.num_of_series = len(xs)
        self.x = np.asarray(xs)

        # store train-validation-test split points 
        self.train_split_size = train_split_size
        self.series_length = self.x[0].shape[1]
        self.num_train = int(self.series_length * train_split_size)
        self.num_cv_train = int(self.num_train * train_split_size)
        self.num_cv_test = self.num_train - self.num_cv_train
        self.num_test = self.series_length - self.num_train

        self.num_of_exogenous_series = self.x[0].shape[0]
        
        self.validation_loss = np.inf

        if ys is None:
            # assume that xs is also 1d
            self.y = np.zeros(self.x.shape[1])
        else:
            self.y = np.asarray(ys)
    
        # model parameters
        self.model_params = dict()

        if params is not None:
            #self.model_params['eta'] = params['eta']
            self.model_params['mu'] = params['mu']
            self.model_params['theta'] = params['theta']
            self.model_params['C'] = params['C']
            self.model_params['c'] = params['c']

        self.l2_param = l2_param
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.num_initializations = num_initializations

        self.verbose = verbose
                    
    def print_log(self, msg):
        if self.verbose is True:
            print(msg)

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

    def predict(self, x, model_params=None):
        """
            Predict the future values of X series given the previous values in
            the series and a list of influential series.

            Parameters
            ----------
            x
                a list of the previous values of the relative sources of influence.
            mode_params
                 model parameters.
        """
        if model_params is None:
            model_params = self.model_params

        predictions = tf.Variable([])
        i = tf.constant(0)
        train_size = tf.shape(x)[1]
        #bias = model_params['eta']

        def loop_body(i, x, pred_history):
            exogenous = tf.reduce_sum(tf.multiply(model_params['mu'], x[:, i]))
            endogenous = model_params['C'] * tf.reduce_sum(
                                                            pred_history *
                                                            tf.pow(self.time_decay_base(tf.shape(pred_history)[0]) + model_params['c'], 
                                                                tf.tile([-1 - model_params['theta']], [tf.shape(pred_history)[0]]))
                                                        )
            new_prediction = tf.add_n([exogenous, endogenous]) 
            pred_history = tf.concat([pred_history, [new_prediction]], axis=0)
            
            i = tf.add(i, 1)
            return [i, x, pred_history]

        loop_condition = lambda i, x, pred_history: tf.less(i, train_size)

        _, _, predictions = tf.while_loop(
                                          cond=loop_condition, 
                                          body=loop_body,
                                          loop_vars=[i, x, predictions], 
                                          shape_invariants=[i.get_shape(), x.get_shape(), tf.TensorShape(None)]
                                         )
        
        return predictions

    def get_test_rmse(self):
        loss = 0

        predictions = self.get_predictions()
        data_length = self.series_length
        
        for i in range(len(predictions)):
            y_truth = self.y[i]
            y_pred = predictions[i]

            loss += np.sqrt(np.sum((y_pred[self.num_train:] - y_truth[self.num_train:]) ** 2))
    
        return loss / (float)(self.num_test)
                
    def train(self, op='adagrad'):
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
        
        for i in range(self.num_initializations):
            
            self.print_log("== Initialization " + str(i + 1))
            loss_value, model_params = self._fit(iteration_number=i,
                                                 optimization_algorithm=op)

            if loss_value < self.validation_loss:
                self.validation_loss = loss_value
                self.model_params = model_params

    def _init_tf_model_variables(self):
        #eta = tf.get_variable('eta', initializer=tf.constant(self.model_params['eta']))                        
        mu = tf.get_variable('mu', initializer=tf.constant(self.model_params['mu']))        
        theta = tf.get_variable('theta', initializer=tf.constant(self.model_params['theta']))  
        C = tf.get_variable('C', initializer=tf.constant(self.model_params['C']))
        c = tf.get_variable('c', initializer=tf.constant(self.model_params['c']))

    def get_predictions(self):
        # predict future values for the test data

        # Instantiate a new model with the trained parameters
        tf.reset_default_graph()
        
        x_observed = tf.placeholder(tf.float32, name='x_observed')

        self._init_tf_model_variables()
        
        pred = self.predict(x_observed)
        predictions = []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.num_of_series):    
                x = self.x[i]
                y = self.y[i]
                
                new_predictions = sess.run(
                                        pred, 
                                        feed_dict={
                                            x_observed: x
                                        }
                                    )
                predictions.append(new_predictions)
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
        y_truth = tf.placeholder(tf.float32, name='y_truth')

        # The model: 
        # sum(mu[i], x_observed[i]) + C * (kernel_base + c ^ -(1 + theta))
        # eta = tf.get_variable(
        #                       name='eta',
        #                       shape=(),
        #                       initializer=tf.random_uniform_initializer(0, 30, seed=RANDOM_SEED + iteration_number)
        #                     )
        mu = tf.get_variable(
                             name='mu',
                             shape=(1, self.num_of_exogenous_series),
                             initializer=tf.random_uniform_initializer(-3, 3, seed=RANDOM_SEED + iteration_number)
                            )        

        theta = tf.get_variable(
                                name='theta',
                                shape=(),
                                initializer=tf.random_uniform_initializer(0, 3, seed=RANDOM_SEED + iteration_number),
                                constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                               )  

        C = tf.get_variable(
                            name='C',
                            shape=(),
                            initializer=tf.random_uniform_initializer(0, 1, seed=RANDOM_SEED + iteration_number),
                            constraint=lambda x: tf.clip_by_value(x, 0.01, np.infty)
                           )

        # c should be non-negative to prevent NaN values in the exponential part
        # e.g.: (-1)^0.5 would return NaN
        c = tf.get_variable(
                            name='c',
                            shape=(),
                            initializer=tf.random_uniform_initializer(0, 5, seed=RANDOM_SEED + iteration_number),
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                           )

        # create params dictionary for easier management
        params_keys = ['mu', 'theta', 'C', 'c']
        params_values = [mu, theta, C, c]
        params = dict(zip(params_keys, params_values))

        pred = self.predict(x_observed, params)
        
        loss = (
                tf.sqrt(tf.reduce_sum(tf.square(y_truth - pred))) + 
                self.l2_param * (tf.reduce_sum(tf.square(mu)) + tf.square(C))
               ) 

        previous_loss = np.inf

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)        
        train_op = optimizer.minimize(loss)
        grad = tf.gradients(loss, [mu, theta, C, c])
        
        validation_loss = 0 

        with tf.Session() as sess:
            tf.set_random_seed(RANDOM_SEED)
            sess.run(tf.global_variables_initializer())
            
            for i in range(self.num_of_series):
                self.print_log("--- Fitting target series #{}".format(i))
                x = self.x[i]
                y = self.y[i]
                
                test_split = int(len(y) * self.train_split_size)
                validation_split = int(test_split * self.train_split_size)
                train_x, train_y = x[:, :self.num_cv_train], y[:self.num_cv_train]
                validation_x, validation_y = x[:, self.num_cv_train:self.num_train], y[self.num_cv_train:self.num_train]
                
                observed_mu, observed_theta, observed_C, observed_c = sess.run([[mu], [theta], [C], [c]])
                observed_loss = sess.run(
                                        [loss],
                                        feed_dict={
                                                    y_truth: train_y,
                                                    x_observed: train_x
                                                }
                                        )
                observed_grad = sess.run(
                                        [grad],
                                        feed_dict={
                                                    y_truth: train_y,
                                                    x_observed: train_x
                                                }
                                        )

                self.print_log(' iter | mu | theta | C | c | validation loss ')
                
                iteration_counter = 1
                while iteration_counter < self.max_iterations: 
                    # gradient step                         
                    sess.run(
                            train_op, 
                            feed_dict={
                                    x_observed: train_x,
                                    y_truth: train_y
                                }
                            )   
                    
                    # get new parameters
                    curr_mu, curr_theta, curr_C, curr_c = sess.run([mu, theta, C, c]) 
                    param_diffs= np.subtract([curr_mu, curr_theta, curr_C, curr_c],
                                            [observed_mu[-1], observed_theta[-1], observed_C[-1], observed_c[-1]])
                    param_diffs_flattened = np.column_stack(param_diffs).ravel()
                    difference_normalized = np.linalg.norm(param_diffs_flattened)
            
                    # update loss
                    curr_loss = sess.run(
                                        loss,
                                        feed_dict={
                                                    x_observed: train_x,
                                                    y_truth: train_y
                                                }
                                        )
                    loss_difference = np.abs(curr_loss - observed_loss[-1])

                    # update gradient
                    curr_grad = sess.run(
                                        grad,
                                        feed_dict={
                                                    x_observed: train_x,
                                                    y_truth: train_y
                                                }
                                        )
                    curr_grad_flattened = np.column_stack(curr_grad).ravel()
                    grad_norm = np.linalg.norm(curr_grad_flattened)

                    # save new values
                    observed_mu.append(curr_mu)
                    observed_theta.append(curr_theta)
                    observed_C.append(curr_C)
                    observed_c.append(curr_c)
                    observed_loss.append(curr_loss)
                    observed_grad.append(curr_grad)
                    
                    if iteration_counter % PRINT_ITERATION_CHECKPOINT_STEPS == 0:
                        self.print_log(
                            ' {} | {} | {} | {} | {} | {}'
                            .format(
                                iteration_counter,
                                curr_mu,
                                curr_theta,
                                curr_C,
                                curr_c,
                                curr_loss
                                )
                            )

                    #check termination conditions
                    if difference_normalized < TOL_PARAM:
                        self.print_log('Parameter convergence in {} iterations!'.format(iteration_counter))
                        break

                    if loss_difference < TOL_LOSS:
                        self.print_log('Loss function convergence in {} iterations!'.format(iteration_counter))
                        break

                    if grad_norm < TOL_GRAD:
                        self.print_log('Gradient convergence in {} iterations!'.format(iteration_counter))
                        break
                    
                    iteration_counter += 1

                validation_loss += sess.run(
                                            loss,
                                            feed_dict={
                                                        x_observed: validation_x,
                                                        y_truth: validation_y
                                                    }
                                        ) 

            params_vals = sess.run([mu, theta, C, c])
            fitted_model_params = dict(zip(params_keys, params_vals)) 
            
        return validation_loss, fitted_model_params
    
    def get_model_parameters(self):
        """
            Getter method to get the model parameters
        """
        return self.model_params.copy()

    def get_test_prediction_error(self):
        predictions = self.get_predictions()

        test_split_start = len(self.train_y) + len(self.validation_y)
        test_preds = predictions[test_split_start:]

        return np.sqrt(np.sum((self.test_y - test_preds) ** 2))

    def plot(self, ax=None):
        predictions = self.get_predictions()[0]
        
        ax = ax or plt.gca()
        ax.plot(self.y[0], 'k--')
        ax.plot(predictions, 'r-')
        return ax
        