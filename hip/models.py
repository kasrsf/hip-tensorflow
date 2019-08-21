import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# disable INFO logs
tf.logging.set_verbosity(tf.logging.ERROR)

RANDOM_SEED = 42
# select the past MEMORY_WINDOW values of prediction when 
# calculating the endogenous influence
MEMORY_WINDOW = 15
class TensorHIP():
    """
        Hawkes Intensity Process Model Implemented and Optimized in TensorFlow
        Used for prediction of time series using the Hawkes Self-Excitation 
        model with one or more exogenous sources of influence
    """
    def __init__(self,verbose=False):
        self.verbose = verbose
        if verbose is True:
            logging.basicConfig(level=logging.INFO)

    def print_log(self, msg):    
        logging.info(msg)

    def time_decay_base(self, i):
        """
            Kernel Base for the time-decaying exponential kernel
            Increasing per each time step in the series

            Parameters
            ----------
            i
                time series length
        """
        return tf.cast(tf.range(i+1, 1, -1), tf.float32)
    # NOTE: IF we set range to end at 1, the endogenous effect becomes exponential since we are always adding the last effect
    # To ways to mitigate. add an offset var c, which given the nonconvexicity of the optimization task, makes training harder
            
    def train(self, exog, y, num_initializations=10,
              l1_param_value=0, l2_param_value=0):
        """
            Fit the best HIP model using multiple random restarts by
            minimizing the loss value of the model on the train data 
        """ 
        # make sure input are ndarrays
        exog = np.asarray(exog, dtype=float)
        y = np.asarray(y, dtype=float)
        best_loss_value = np.inf
        best_model_params = None
        for i in range(num_initializations):
            self.print_log("== Initialization " + str(i + 1))
            learned_params, train_loss = self._fit(exog, y, initialization_number=i,
                                                   l1_param=l1_param_value,
                                                   l2_param=l2_param_value)
            if train_loss < best_loss_value:
                best_loss_value = train_loss
                best_model_params = learned_params
        self.model_params = best_model_params
        self.model_params['l1-alpha'] = l1_param_value
        self.model_params['l2-alpha'] = l2_param_value
        
    def _fit(self, exog, y, initialization_number, 
             l1_param, l2_param, max_iterations=500):
        """
            Internal method for fitting the model at each iteration of the
            training process
        """
        tf.reset_default_graph()
        exog_observed = tf.placeholder(tf.float32, name='exog_observed')
        y_truth = tf.placeholder(tf.float32, name='y_truth')
        params = self._init_tf_model_variables(exog_series_count=len(exog), 
                                               random_seed=RANDOM_SEED + initialization_number)
        pred = self._predict(exog_observed, params)
        loss = (
            (tf.sqrt(tf.reduce_mean(tf.square(y_truth - pred)))) +
            (l1_param * tf.reduce_mean(tf.abs(params['mu']))) +
            (l1_param * tf.reduce_mean(tf.abs(params['eta']))) +
            (l2_param * (tf.reduce_mean(tf.square(params['mu'])))) +
            (l2_param * (tf.reduce_mean(tf.square(params['eta']))))
        ) 
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
                                                            loss, 
                                                            method='L-BFGS-B',
                                                            options={'maxiter': max_iterations}
                                                        )             
        validation_loss_sum = 0 
        with tf.Session() as sess:
            tf.set_random_seed(RANDOM_SEED)
            sess.run(tf.global_variables_initializer())            
            optimizer.minimize(session=sess,
                                feed_dict={
                                    exog_observed: exog,
                                    y_truth: y
                                }
                            )
            train_loss = sess.run(
                                    loss,
                                    feed_dict={
                                                exog_observed: exog,
                                                y_truth: y
                                            }
                                )    
            learned_params = sess.run([params])[0]
        return learned_params, train_loss

    def _init_tf_model_variables(self, exog_series_count=0, random_seed=RANDOM_SEED):
        # use trained params in case model is trained
        # else, we won't have the model_params attribute and will 
        # randomize the initial values
        try:
            mu = tf.get_variable('mu', initializer=tf.constant(self.model_params['mu']))
            eta = tf.get_variable('eta', initializer=tf.constant(self.model_params['eta']))                                               
            theta = tf.get_variable('theta', initializer=tf.constant(self.model_params['theta']))  
            C = tf.get_variable('C', initializer=tf.constant(self.model_params['C']))        
        except AttributeError:
            tf.set_random_seed(random_seed)
            mu = tf.get_variable(name='mu', 
                                 shape=(1, exog_series_count),
                                 initializer=tf.random_normal_initializer(mean=1, 
                                                                          stddev=1,
                                                                          seed=random_seed)
            )
            eta = tf.get_variable(name='eta',
                shape=(), initializer=tf.random_normal_initializer(mean=0, stddev=0.5)
            )
            # enforce positivity constraint on theta and C to avoid exponential corner cases  
            theta = tf.get_variable(name='theta',
                shape=(), initializer=tf.random_normal_initializer(mean=10, stddev=5),
                constraint=lambda x: tf.clip_by_value(x, 0.5, np.infty),
            )
            C = tf.get_variable(name='C',
                shape=(),
                initializer=tf.random_normal_initializer(mean=3, stddev=1),
                constraint=lambda x: tf.clip_by_value(x, 0.01, np.infty),
            )
        #    if self.fixed_eta is True:
        #        eta = tf.constant(self.model_params['eta'])
        # if self.fixed_theta is True:
        #     theta = tf.constant(self.model_params['theta'])
        # if self.fixed_C is True:
        #     C = tf.constant(self.model_params['C'])            
        # if 'c' in self.model_params:
        #     if self.fixed_c is True:
        #         c = tf.constant(self.model_params['c'])
        #     else:
        #         c = tf.get_variable('c', initializer=tf.constant(self.model_params['c']))
        # else:
        #     c = tf.get_variable(
        #         name='c',
        #         shape=(),
        #         initializer=tf.random_normal_initializer(mean=1, stddev=1),
        #         constraint=lambda x: tf.clip_by_value(x, 0, np.infty),
        #     )
        return {'eta': eta, 'mu': mu, 'theta': theta, 'C': C}
    
    def _predict(self, x, model_params):
        """
            Predict the future values of X series given the previous values in
            the series and a list of influential exogenous series.

            Parameters
            ----------
            x
                a list of the previous values of the relative sources of influence.
            mode_params
                 model parameters. contains ['eta', 'mu', 'theta', 'C']
        """
        predictions = tf.Variable([])
        i = tf.constant(0)
        train_size = tf.shape(x)[1]
        bias = model_params['eta']
        def loop_body(i, x, pred_history):
            exogenous = tf.reduce_sum(tf.multiply(model_params['mu'], x[:, i]))
            endo_history_window_start = tf.maximum(0, i - MEMORY_WINDOW)
            endo_history = pred_history[endo_history_window_start:]
            endogenous = model_params['C'] * tf.reduce_sum(endo_history *
                                                           tf.pow(self.time_decay_base(i - endo_history_window_start) + tf.constant(0.01),#, model_params['c']), 
                                                                tf.tile([-1 - model_params['theta']], [i - endo_history_window_start]))
                                                        )
            new_prediction = tf.add_n([bias
                                       , exogenous
                                       , endogenous]) 
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

    def get_predictions(self, exog, start=0):
        # make sure data is in valid format
        exog = np.asarray(exog, dtype=float)
        # predict future values for the input data
        # Instantiate a new model with the trained parameters
        tf.reset_default_graph()
        exog_observed = tf.placeholder(tf.float32, name='exog_observed')
        params = self._init_tf_model_variables()
        pred = self._predict(exog_observed, params)
        predictions = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            predictions = sess.run(pred, feed_dict={exog_observed: exog})
        return predictions[start:]
    
    def get_model_parameters(self):
        """
            Getter method to get the model parameters
        """
        return self.model_params.copy()