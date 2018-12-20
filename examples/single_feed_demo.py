import pandas as pd
import sys

from code.utils import load_data_from_csv
from hip.models import TensorHIP

if __name__ == '__main__':
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = 'data/sample_data_1.csv'

    print("Reading File: {}".format(filename))
    features, target, feature_names, target_name = load_data_from_csv(filename=filename)

    hip_model = TensorHIP(xs=[features],
                  ys=[target],    
                  l1_param=0.1, 
                  feature_names=feature_names,
                  scale_series=True)
    hip_model.train()    

    print("Model Parameters= ", hip_model.get_model_parameters())
    print("Learned Feature Weights = ", hip_model.get_weights_dict())

    hip_model.plot()