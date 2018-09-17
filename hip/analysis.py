import numpy as np

def compute_rmse_error(cost_vector):
    """
        returns the RMSE error on the test data
    """
    return np.sqrt(cost_vector.sum() / len(cost_vector))
