import numpy as np
import pickle

from hip.models import TensorHIP

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    series_1_xs = [daily_share]
    series_1_y = daily_view 
    
    MAXIMUM_NOISE = 250
    random_noise = (np.random.rand(len(daily_view)) * MAXIMUM_NOISE) - (MAXIMUM_NOISE / 2)
    series_2_y = daily_view + random_noise
    series_2_xs = [daily_share]

    MAXIMUM_NOISE = 500
    random_noise = (np.random.rand(len(daily_view)) * MAXIMUM_NOISE) - (MAXIMUM_NOISE / 2)
    series_3_y = series_2_y + random_noise
    series_3_xs = [daily_share]

    MAXIMUM_NOISE = 750
    random_noise = (np.random.rand(len(daily_view)) * MAXIMUM_NOISE) - (MAXIMUM_NOISE / 2)
    series_4_y = series_3_y + random_noise
    series_4_xs = [daily_share]

    model = TensorHIP([series_1_xs, series_2_xs, series_3_xs, series_4_xs], [series_1_y, series_2_y, series_3_y, series_4_y])
    model.train(num_iterations=3)
    print(model.get_model_parameters())
    model.plot_predictions()

    