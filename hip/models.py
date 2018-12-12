import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug


from hip.utils import TimeSeriesScaler

# stop the optimization process after doing a certain number of iterations
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-4, 1e-4, 1e-4
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
                 l1_param=0,
                 l2_param=0,
                 learning_rate=0.5,
                 num_initializations=3,
                 initalization_method='normal',
                 max_iterations=100,
                 params=None,
                 fix_c_param_value=None,
                 fix_theta_param_value=None,
                 fix_C_param_value=1.0,
                 scale_series=True,
                 verbose=False,
                 optimizer='l-bfgs'
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
        self.fixed_c = False
        if fix_c_param_value is not None:
            self.fixed_c = True
            self.model_params['c'] = fix_c_param_value
        self.fixed_theta = False
        if fix_theta_param_value is not None:
            self.fixed_theta = True
            self.model_params['theta'] = fix_theta_param_value
        self.fixed_C = False
        if fix_C_param_value is not None:
            self.fixed_C = True
            self.model_params['C'] = fix_C_param_value

        if params is not None:
            self.model_params['eta'] = params['eta']
            self.model_params['mu'] = params['mu']
            self.model_params['theta'] = params['theta']
            self.model_params['C'] = params['C']
            self.model_params['c'] = params['c']

        self.l1_param = l1_param
        self.l2_param = l2_param
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.num_initializations = num_initializations
        self.initalization_method = initalization_method

        self.verbose = verbose
        self.scale_series = scale_series
        if scale_series is True:
            self.series_scaler = TimeSeriesScaler()

        self.optimizer = optimizer

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
        return tf.cast(tf.range(i, tf.maximum(i - 7, 0), -1), tf.float32)

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
        bias = model_params['eta']
        def loop_body(i, x, pred_history):
            exogenous = tf.reduce_sum(tf.multiply(model_params['mu'], x[:, i]))
            endogenous = model_params['C'] * tf.reduce_sum(
                                                            pred_history[tf.maximum(i-7, 0):] *
                                                            tf.pow(self.time_decay_base(tf.shape(pred_history)[0]) + tf.maximum(0.0, model_params['c']), 
                                                                tf.tile([-1 - model_params['theta']], [tf.minimum(7, tf.shape(pred_history)[0])]))
                                                        )
            new_prediction = tf.add_n([bias, exogenous, endogenous]) 
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
            
    def train(self):
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
        best_validation_loss = self.validation_loss       
        for i in range(self.num_initializations):
            
            self.print_log("== Initialization " + str(i + 1))
            loss_value, model_params = self._fit(iteration_number=i, op=self.optimizer)

            if loss_value < best_validation_loss:
                best_validation_loss = loss_value
                best_model_params = model_params
 
        self.validation_loss = best_validation_loss
        self.model_params = best_model_params

    def _init_tf_model_variables(self, random_seed=RANDOM_SEED):
        if 'eta' in self.model_params:
            eta = tf.get_variable('eta', initializer=tf.constant(self.model_params['eta']))                        
        else:
            if self.initalization_method == 'uniform':
                eta = tf.get_variable(
                                name='eta',
                                shape=(),
                                initializer=tf.random_uniform_initializer(0, 30, seed=random_seed)
                                )
            elif self.initalization_method == 'normal':
                eta = tf.get_variable(
                                name='eta',
                                shape=(),
                                initializer=tf.random_normal_initializer(mean=15, stddev=5, seed=random_seed)
                                )
        
        if 'mu' in self.model_params:
            mu = tf.get_variable('mu', initializer=tf.constant(self.model_params['mu']))        
        else:
            if self.initalization_method == 'uniform':
                mu = tf.get_variable(
                                name='mu',
                                shape=(1, self.num_of_exogenous_series),
                                initializer=tf.random_uniform_initializer(-3, 3, seed=random_seed)
                                )    
            elif self.initalization_method == 'normal':
                mu = tf.get_variable(
                                name='mu',
                                shape=(1, self.num_of_exogenous_series),
                                initializer=tf.random_normal_initializer(mean=0, stddev=3, seed=random_seed)
                                )
            

        if 'theta' in self.model_params:
            if self.fixed_theta is True:
                theta = tf.constant(self.model_params['theta'])
            else:
                theta = tf.get_variable('theta', initializer=tf.constant(self.model_params['theta']))  
        else:
            if self.initalization_method == 'uniform':
                theta = tf.get_variable(
                            name='theta',
                            shape=(),
                            initializer=tf.random_uniform_initializer(0, 3, seed=random_seed),
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                            )  
            elif self.initalization_method == 'normal':
                theta = tf.get_variable(
                            name='theta',
                            shape=(),
                            initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=random_seed),
                            constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                            )  

        if 'C' in self.model_params:
            if self.fixed_C is True:
                C = tf.constant(self.model_params['C'])
            else:
                C = tf.get_variable('C', initializer=tf.constant(self.model_params['C']))
        else:
            if self.initalization_method == 'uniform':
                C = tf.get_variable(
                        name='C',
                        shape=(),
                        initializer=tf.random_uniform_initializer(0, 1, seed=random_seed),
                        constraint=lambda x: tf.clip_by_value(x, 0.01, np.infty)
                    )
            elif self.initalization_method == 'normal':
                C = tf.get_variable(
                        name='C',
                        shape=(),
                        initializer=tf.random_normal_initializer(mean=0, stddev=1, seed=random_seed),
                        constraint=lambda x: tf.clip_by_value(x, 0.01, np.infty)
                    )
        
        if 'c' in self.model_params:
            if self.fixed_c is True:
                c = tf.constant(self.model_params['c'])
            else:
                c = tf.get_variable('c', initializer=tf.constant(self.model_params['c']))
        else:
            if self.initalization_method == 'uniform':
                c = tf.get_variable(
                        name='c',
                        shape=(),
                        initializer=tf.random_uniform_initializer(0, 1, seed=random_seed),
                        constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                    )
            elif self.initalization_method == 'normal':
                c = tf.get_variable(   
                        name='c',
                        shape=(),
                        initializer=tf.random_normal_initializer(mean=0, stddev=5, seed=random_seed),
                        constraint=lambda x: tf.clip_by_value(x, 0, np.infty)
                    )

        return {
                'eta': eta,
                'mu': mu, 
                'theta': theta, 
                'C': C, 
                'c': c
        }

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
                if self.scale_series is True:
                    x = self.series_scaler.transform_x(self.x[i])
                else:
                    x = self.x[i]
                
                new_predictions = sess.run(
                                        pred, 
                                        feed_dict={
                                            x_observed: x
                                        }
                                    )
                predictions.append(new_predictions)

        if self.scale_series is True:
            return self.series_scaler.invert_transform_ys(predictions)
        else:
            return predictions
   
    def _fit(self, iteration_number, op='adam'):
        """
            Internal method for fitting the model at each iteration of the
            training process
        """
        tf.reset_default_graph()
        x_observed = tf.placeholder(tf.float32, name='x_observed')
        y_truth = tf.placeholder(tf.float32, name='y_truth')

        # The model: 
        # sum(mu[i], x_observed[i]) + C * (kernel_base + c ^ -(1 + theta))
          
        # create params dictionary for easier management
        params_keys = ['eta', 'mu', 'theta', 'C', 'c']
        params = self._init_tf_model_variables(random_seed=RANDOM_SEED + iteration_number)
        eta = params['eta']
        mu = params['mu']
        theta = params['theta']
        C = params['C']
        c = params['c']

        pred = self.predict(x_observed, params)
        loss = (
                tf.sqrt(tf.reduce_sum(tf.square(y_truth - pred))) + 
                self.l1_param * (tf.reduce_sum(tf.abs(mu)) + tf.abs(C)) + 
                self.l2_param * (tf.reduce_sum(tf.square(mu)) + tf.square(C))
               ) 
        previous_loss = np.inf
        if op == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)        
            train_op = optimizer.minimize(loss)
        elif op == 'l-bfgs':
            optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, 
                                                               method='L-BFGS-B',
                                                               options={'maxiter': self.max_iterations}
                                                            )            
        
        grad = tf.gradients(loss, [mu, theta, C, c])
        validation_loss = 0 
        self.losses = []
        with tf.Session() as sess:
            tf.set_random_seed(RANDOM_SEED)
            sess.run(tf.global_variables_initializer())
            
            params_vals = sess.run([eta, mu, theta, C, c])
            fitted_model_params = dict(zip(params_keys, params_vals)) 

            if self.scale_series is True:
                xs = self.series_scaler.transform_xs(self.x)
                ys = self.series_scaler.transform_ys(self.y)
            else:
                xs = self.x
                ys = self.y

            for i in range(self.num_of_series):
                self.print_log("--- Fitting target series #{}".format(i + 1))
                x = xs[i]
                y = ys[i]
                
                test_split = int(len(y) * self.train_split_size)
                validation_split = int(test_split * self.train_split_size)
                train_x, train_y = x[:, :self.num_cv_train], y[:self.num_cv_train]
                validation_x, validation_y = x[:, self.num_cv_train:self.num_train], y[self.num_cv_train:self.num_train]
                    
                if op == 'adam':
                    observed_eta, observed_mu, observed_theta, observed_C, observed_c = sess.run([[eta], [mu], [theta], [C], [c]])
                    new_predictions = sess.run(
                                            pred, 
                                            feed_dict={
                                                x_observed: x
                                            }
                                        )
                    
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

                    self.print_log(' iter | eta | mu | theta | C | c | validation loss ')
                    
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
                        curr_eta, curr_mu, curr_theta, curr_C, curr_c = sess.run([eta, mu, theta, C, c]) 
                        param_diffs= np.subtract([curr_mu, curr_eta, curr_theta, curr_C, curr_c],
                                                [observed_mu[-1], observed_eta[-1], observed_theta[-1], 
                                                observed_C[-1], observed_c[-1]])
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
                        self.losses.append(curr_loss)
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
                        observed_eta.append(curr_eta)
                        observed_mu.append(curr_mu)
                        observed_theta.append(curr_theta)
                        observed_C.append(curr_C)
                        observed_c.append(curr_c)
                        observed_loss.append(curr_loss)
                        observed_grad.append(curr_grad)
                        
                        if iteration_counter % PRINT_ITERATION_CHECKPOINT_STEPS == 0:
                            self.print_log(
                                ' {} | {} | {} | {} | {} | {} | {}'
                                .format(
                                    iteration_counter,
                                    curr_eta,
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
                elif op == 'l-bfgs':
                    optimizer.minimize(
                                        session=sess,
                                        feed_dict={
                                            x_observed: train_x,
                                            y_truth: train_y
                                        }
                    )

                validation_loss += sess.run(
                                            loss,
                                            feed_dict={
                                                        x_observed: validation_x,
                                                        y_truth: validation_y
                                                    }
                                        ) 

            params_vals = sess.run([eta, mu, theta, C, c])
            fitted_model_params = dict(zip(params_keys, params_vals)) 
            
        return validation_loss, fitted_model_params
    
    def get_model_parameters(self):
        """
            Getter method to get the model parameters
        """
        return self.model_params.copy()

    def get_validation_rmse(self):
        predictions = self.get_predictions()
        validation_split_start = self.num_cv_train
        validation_split_end = self.num_train

        error = 0
        for i in range(len(predictions)):
            y_truth = self.y[i][validation_split_start:validation_split_end]
            y_pred = predictions[i][validation_split_start:validation_split_end]
            error += np.sum(y_pred - y_truth) ** 2 / len(y_truth)

        return np.sqrt(error / len(predictions))

    def get_test_rmse(self):
        predictions = self.get_predictions()
        test_split_start = self.num_train 

        error = 0
        for i in range(len(predictions)):
            y_truth = self.y[i][test_split_start:]
            y_pred = predictions[i][test_split_start:]
            error += np.sum(y_pred - y_truth) ** 2 / len(y_truth)

        return np.sqrt(error / len(predictions))

    def plot(self, ax=None):
        predictions = self.get_predictions()
        
        num_of_series = len(predictions)
        data_length = len(predictions[0])
        data_test_split_point = self.num_train

        srows = (int)(np.ceil(np.sqrt(num_of_series)))

        display_plot = False

        if ax is None:
            fig, axes = plt.subplots(srows, srows, sharex='all')

        for i in range(num_of_series):
            row = (int)(i / srows)
            col = (int)(i % srows)
            truth = self.y[i]
            pred = predictions[i]

            if ax is None:
                display_plot = True
                if num_of_series == 1:
                    ax = plt
                else:
                    ax = axes[row, col]
    
            ax.axvline(data_test_split_point, color='k')
            ax.plot(np.arange(data_length), truth, 'k--', label='Observed #views')

            # plot predictions on training data with a different alpha to make the plot more clear            
            ax.plot(
                        np.arange(data_test_split_point+1),
                        pred[:data_test_split_point+1], 
                        'b-',
                        alpha=0.5,
                        label='Model Fit'
                    )
            ax.plot(
                        np.arange(data_test_split_point, data_length),
                        pred[data_test_split_point:], 
                        'b-',
                        alpha=1,
                        label='Model Predictions'
                    )
        
        if display_plot is True:
            fig.show()