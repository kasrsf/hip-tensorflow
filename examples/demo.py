import numpy as np
import pickle

from hip.models import TensorHIP

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    series_1_xs = [daily_share, daily_share]
    series_1_y = daily_view

    xs = [series_1_xs]
    ys = [series_1_y] 
    
    model = TensorHIP(
                      xs, ys,
                      feature_names=['random feature 1', 'random feature 2'],
                      eta_param_mode="constant",
                      fix_eta_param_value=0.5
                    )
    model.train()
    
    print(model.get_model_parameters())
    print(model.get_test_rmse())
    print(model.get_weights_dict())
    model.plot()
    