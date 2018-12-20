import csv  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data_from_csv(filename):
    raw_data_df = pd.read_csv(filename)
    # always assume that the last column in the CSV file is the target series
    # and the rest are time-series data for the features
    features, target = np.split(raw_data_df, [-1], axis=1) 
    feature_names = list(features)
    target_name = list(target)[0]
    return features.values.T, target.values.T[0], feature_names, target_name

def print_params_to_tsv(params, feature_name):
    eta = params['eta']
    mu = params['mu'][0][0]
    c = params['c']
    theta = params['theta']

    param_names = ['eta', 'mu', 'c', 'theta']
    param_values = [eta, mu, c, theta]

    print('\t'.join([str(x) for x in param_names]))
    print('\t'.join([str(x) for x in param_values]))

def plot_predictions(y_truth, y_predictions, xs=None, train_test_split_point=0.8, legend=True):
        """
            Plot the current predictions from the fitted model 
        """
        num_of_series = len(y_truth)
        data_length = len(y_truth[0])
        data_test_split_point = (int)(data_length * train_test_split_point)

        srows = (int)(np.ceil(np.sqrt(num_of_series)))

        fig, axes = plt.subplots(srows, srows, sharex='all')
        for i in range(num_of_series):
            row = (int)(i / srows)
            col = (int)(i % srows)

            truth = y_truth[i]
            pred = y_predictions[i]

            if num_of_series == 1:
                ax = plt
            else:
                ax = axes[row, col]

            ax.axvline(data_test_split_point, color='k')
            ax.plot(np.arange(data_length), truth, 'k--', label='Observed #views')

            if xs is not None:
                x = xs[i]
                
                colors = iter(plt.cm.rainbow(np.linspace(0, 1, len(x))))
                for index, exo_source in enumerate(x):
                    c = next(colors)
                    ax.plot(np.arange(data_length), exo_source, c=c, alpha=0.3)

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

        plt.show()

def get_test_rmse(truth, predictions, train_test_split=0.8):
        loss = 0
        split_point = (int)(train_test_split * len(truth[0])) + 1

        for i in range(len(predictions)):
            y_truth = truth[i][split_point:]
            y_pred = predictions[i][split_point:]

            loss += np.sqrt(np.sum(y_pred - y_truth) ** 2) / len(y_truth)
    
        return loss

class TimeSeriesScaler():
    def __init__(self):
        self.y_mins = []
        self.y_maxs = []

    def transform_x(self, x):
        x_min = np.min(x)
        x_max = np.max(x)
        return (x - x_min) / (x_max - x_min)

    def transform_xs(self, xs):
        scaled_xs = []
        for x_series in xs:
            scaled_x_series = []
            for x in x_series:
                scaled_x = self.transform_x(x)

                scaled_x_series.append(scaled_x)
            scaled_xs.append(scaled_x_series)

        return np.asarray(scaled_xs)
    
    def transform_add_y(self, y):
        y_min = np.min(y)
        y_max = np.max(y)

        scaled_y = (y - y_min) / (y_max - y_min)

        self.y_mins.append(y_min)
        self.y_maxs.append(y_max)

        return scaled_y

    def transform_ys(self, ys):
        self.y_mins = []
        self.y_maxs = []

        scaled_ys = []
        for y in ys:
            scaled_y = self.transform_add_y(y)

            scaled_ys.append(scaled_y)

        return np.asarray(scaled_ys)

    def invert_transform_ys(self, scaled_ys):
        rescaled_ys = []
        for index, scaled_y in enumerate(scaled_ys):
            rescaled_y = (
                            scaled_y * (self.y_maxs[index] - self.y_mins[index]) +
                            self.y_mins[index]
                        )  

            rescaled_ys.append(rescaled_y)

        return np.asarray(rescaled_ys)