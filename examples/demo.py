import numpy as np
import pickle

from hip.models import TensorHIP
from hip.utils import plot_predictions, TimeSeriesScaler

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    series_1_xs = [daily_share, daily_share]
    series_1_y = daily_view

    xs = [series_1_xs, series_1_xs]
    ys = [series_1_y, series_1_y] 
    ts = TimeSeriesScaler()
    scaled_xs = ts.transform_x(xs)
    scaled_ys = ts.transform_y(ys)
    model = TensorHIP(
                      scaled_xs,
                      scaled_ys,
                      l2_param=0.1
                    )
    model.train(num_iterations=1)
    print(model.get_model_parameters())
    print(model.get_test_rmse())
    scaled_preds = model.get_predictions()
    rescaled_preds = ts.invert_transform_y(scaled_preds)
    plot_predictions(y_truth=ys, y_predictions=rescaled_preds)
    #model.plot_predictions()

    