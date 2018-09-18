import os
import json
import bz2
import pickle

from hip.models import TensorHIP

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    num_train = 90
    num_test = 30

    model = TensorHIP([daily_share], daily_view, num_train, num_test)
    model.fit(num_iterations=5)
    model.plot_predictions()