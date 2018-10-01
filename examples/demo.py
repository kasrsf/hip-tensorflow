import pickle

from hip.models import TensorHIP

# Data and test code from andrei rizoiu's code https://github.com/andrei-rizoiu/hip-popularity
if __name__ == '__main__':
    # Load the Youtube view/share data for a random video ('X0ZEt_GZfkA')
    # from the active dataset @ https://github.com/andrei-rizoiu/hip-popularity
    daily_share, daily_view, _ = pickle.load(open('../data/views.p', 'rb'))
    
    model = TensorHIP([daily_share], daily_view)
    model.train(num_iterations=1)
    print(model.get_model_parameters())
    model.plot_predictions()

    