import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# stop the optimization process after doing a certain number of iterations
OPTIMIZAITION_MAX_ITERATIONS = 1000
TOL_PARAM, TOL_LOSS, TOL_GRAD = 1e-8, 1e-8, 1e-8
PRINT_ITERATION_CHECKPOINT_STEPS = 20

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
                 x,
                 y=None,
                 train_split_size=0.8,
                 params=None
                ):
        self.x = np.asarray(x)
        if y is None:
            self.y = np.zeros(self.x.shape[1])
        else:
            self.y = np.array(y)

        # do the train-validation-test split 
        # (use same split: Train = 0.8 * 0.8 * length, 
        # validation = 0.2 * 0.8 * length, test = 0.2 * length)
        test_split = int(len(self.y) * train_split_size)
        validation_split = int(test_split * train_split_size)
        self.train_x, self.train_y = self.x[:, :validation_split], self.y[:validation_split]
        self.validation_x, self.validation_y = (
                                                  self.x[:, validation_split:test_split], 
                                                  self.y[validation_split:test_split]
                                               )  
        self.test_x, self.test_y = self.x[:, test_split:], self.y[test_split:]
        # model parameters
        #self.gamma = 0
        self.model_params = dict()

        if params is not None:
            #self.model_params['eta'] = params['eta']
            self.model_params['mu'] = params['mu']
            self.model_params['theta'] = params['theta']
            self.model_params['C'] = params['C']
            self.model_params['c'] = params['c']
                    
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

    def predict(self, x, model_params=None,se=None):
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
                
    def train(self, num_iterations, op='adagrad', verbose=True, regularizer=None):
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
            loss_value, model_params = self._fit(iteration_number=i,
                                                 optimization_algorithm=op,
                                                 regularizer=regularizer)

            if loss_value < best_loss:
                best_loss = loss_value
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

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            predictions = sess.run(
                                    pred, 
                                    feed_dict={
                                        x_observed: self.x
                                    }
                                  )

        # TODO: What to do when predictions are zero (enforce max(0, pred)?)
        # for i in range(len(predictions)):
        #     if predictions[i] < 0:
        #         predictions[i] = 0
                
        return predictions
   
    def _fit(self, iteration_number, optimization_algorithm='adagrad', regularizer=None):
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
                             shape=(1, len(self.x)),
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
        
        # TODO: Check effect of adding regularization
        #if regularizer is None:
        loss = tf.sqrt(tf.reduce_sum(tf.square(y_truth - pred)))
        #elif regularizer == 'l2':
        #    loss = tf.sqrt(tf.reduce_sum(tf.square(y_truth - pred))) + tf.reduce_sum(tf.square(mu)) + tf.square(C)

        previous_loss = np.inf
        iteration_counter = 1

        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)        
        train_op = optimizer.minimize(loss)
        grad = tf.gradients(loss, [mu, theta, C, c])
        
        with tf.Session() as sess:
            tf.set_random_seed(RANDOM_SEED)
            sess.run(tf.global_variables_initializer())
            
            observed_mu, observed_theta, observed_C, observed_c = sess.run([[mu], [theta], [C], [c]])

            observed_loss = sess.run(
                                    [loss],
                                    feed_dict={
                                                y_truth: self.train_y,
                                                x_observed: self.train_x
                                            }
                                    )
            observed_grad = sess.run(
                                    [grad],
                                    feed_dict={
                                                y_truth: self.train_y,
                                                x_observed: self.train_x
                                            }
                                    )
        
            print(' iter | mu | theta | C | c | loss ')
        
            while iteration_counter < OPTIMIZAITION_MAX_ITERATIONS: 
                # gradient step                         
                sess.run(
                        train_op, 
                        feed_dict={
                                x_observed: self.train_x,
                                y_truth: self.train_y
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
                                                x_observed: self.train_x,
                                                y_truth: self.train_y
                                            }
                                    )
                loss_difference = np.abs(curr_loss - observed_loss[-1])

                # update gradient
                curr_grad = sess.run(
                                    grad,
                                    feed_dict={
                                                x_observed: self.train_x,
                                                y_truth: self.train_y
                                            }
                                    )
                curr_grad_flattened = np.column_stack(curr_grad).ravel()
                grad_norm = np.linalg.norm(curr_grad_flattened)

                # save new values
                observed_mu.append(curr_mu)
                observed_theta.append(curr_theta)
                observed_C.append(curr_C)
                observed_theta.append(curr_c)
                observed_loss.append(curr_loss)
                observed_grad.append(curr_grad)
                
                if iteration_counter % PRINT_ITERATION_CHECKPOINT_STEPS == 0:
                    print(' {} | {} | {} | {} | {} | {}'
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
                    print('Parameter convergence in {} iterations!'.format(iteration_counter))
                    break

                if loss_difference < TOL_LOSS:
                    print('Loss function convergence in {} iterations!'.format(iteration_counter))
                    break

                if grad_norm < TOL_GRAD:
                    print('Gradient convergence in {} iterations!'.format(iteration_counter))
                    break
                
                iteration_counter += 1
            
            prds = sess.run(
                    pred,
                    feed_dict={
                                x_observed: self.train_x,
                                y_truth: self.train_y
                            }
                    )

            validation_loss = sess.run(
                                        loss,
                                        feed_dict={
                                                    x_observed: self.validation_x,
                                                    y_truth: self.validation_y
                                                }
                                    ) 

            params_vals = sess.run([mu, theta, C, c])
            fitted_model_params = dict(zip(params_keys, params_vals)) 
    
        return validation_loss, fitted_model_params
    
    def plot_predictions(self, legend=True):
        """
            Plot the current predictions from the fitted model 
        """
        predictions = self.get_predictions()
        
        data_length = len(self.y)
        data_test_split_point = len(self.train_y) + len(self.validation_y)

        plt.figure(figsize=(8, 8))
        plt.axvline(data_test_split_point, color='k')
        plt.plot(np.arange(data_length), self.y, 'k--', label='Observed #views')

        colors = iter(plt.cm.rainbow(np.linspace(0, 1, self.x.shape[0])))
        for index, exo_source in enumerate(self.x):
            c = next(colors)
            plt.plot(np.arange(data_length), exo_source, c=c, alpha=0.3)

        # plot predictions on training data with a different alpha to make the plot more clear            
        plt.plot(
                    np.arange(data_test_split_point+1),
                    predictions[:data_test_split_point+1], 
                    'b-',
                    alpha=0.5,
                    label='Model Fit'
                )
        plt.plot(
                    np.arange(data_test_split_point, data_length),
                    predictions[data_test_split_point:], 
                    'b-',
                    alpha=1,
                    label='Model Predictions'
                )

        if legend:
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

    def get_test_prediction_error(self):
        predictions = self.get_predictions()

        test_split_start = len(self.train_y) + len(self.validation_y)
        test_preds = predictions[test_split_start:]

        return np.sqrt(np.sum((self.test_y - test_preds) ** 2))