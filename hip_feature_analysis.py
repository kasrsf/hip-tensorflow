from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

from hip.models import TensorHIP
from hip.utils import load_data_from_csv, save_params_to_tsv

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        feature_index = int(sys.argv[2])
    else:
        input_path = 'data/sample_data_2/'
        feature_index = 0 

    input_files = []
    xs = []
    ys = []
    for f in listdir(input_path):
        file_path = join(input_path, f)
        if isfile(file_path) and file_path.lower().endswith('.csv'):
            input_files.append(file_path)
            features, target, feature_names, target_name = load_data_from_csv(file_path)

            xs.append([features[feature_index]])
            ys.append(target)
    hip_model = TensorHIP(xs=xs,
                  ys=ys,    
                  feature_names=feature_names,
                  verbose=True)
    hip_model.train()    

    model_params =  hip_model.get_model_parameters()
    save_params_to_tsv(params=model_params, feature_name=feature_names[feature_index])