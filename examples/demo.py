import numpy as np
import pickle

from hip.models import TensorHIP
from hip.utils import FeatureScalingNormalizer

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    series_1_xs = [daily_share]
    series_1_y = daily_view 

    model = TensorHIP(
                      [series_1_xs],
                      [series_1_y],
                      l2_param=0.1
                    )
    model.train(num_iterations=1)
    print(model.get_model_parameters())
    print(model.get_test_rmse())
    #model.plot_predictions()

    