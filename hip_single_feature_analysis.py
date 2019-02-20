from os import listdir
from os.path import isfile, join
import pandas as pd
import sys
import time

from hip.models import TensorHIP
from hip.utils import load_data_from_csv, print_params_to_tsv

if __name__ == '__main__':
    sys.stderr.write("loading the files\n")
    sys.stderr.flush()

    if len(sys.argv) == 3:
        input_path = sys.argv[1]
        feature_index = int(sys.argv[2])
    else:
        raise SyntaxError("Insufficient arguments")

    input_files = []
    xs = []
    ys = []
    file_paths = []
    for f in listdir(input_path):
        file_path = join(input_path, f)
        if isfile(file_path) and file_path.lower().endswith('.csv'):
            input_files.append(file_path)
            features, target, feature_names, target_name = load_data_from_csv(
                file_path)
            xs.append([features[feature_index]])
            input_feature_names = [feature_names[feature_index]]
            ys.append(target)
            file_paths.append(file_path)
    sys.stderr.write("beginning the training\n")
    sys.stderr.flush()

    start_time = time.time()

    # theta param value fixed (to 4)
    hip_model = TensorHIP(xs=xs, ys=ys,
                          feature_names=input_feature_names,
                          num_initializations=1,
                          fix_theta_param_value=4,
                          verbose=False)

    # eta default random initialization
    # hip_model = TensorHIP(xs=xs, ys=ys,
    #               feature_names=input_feature_names,
    #               num_initializations=1,
    #               verbose=False)

    # eta fixed to the average value of exogenous signal
    # hip_model = TensorHIP(xs=xs, ys=ys,
    #               eta_param_mode='exo_mean',
    #               feature_names=feature_names,
    #               num_initializations=1,
    #               verbose=False)

    # eta fixed to the average value of target signal
    # hip_model = TensorHIP(xs=xs, ys=ys,
    #               eta_param_mode='target_mean',
    #               feature_names=feature_names,
    #               num_initializations=1,
    #               verbose=False)

    # eta fixed to the average value of target signal
    # hip_model = TensorHIP(xs=xs, ys=ys,
    #               eta_param_mode='constant',
    #               fix_eta_param_value=0.1,
    #               feature_names=feature_names,
    #               num_initializations=1,
    #               verbose=False)
    hip_model.train()

    sys.stderr.write("\ntraining completed in {} seconds\n".format(
        time.time() - start_time))

    model_params = hip_model.get_model_parameters()
    print_params_to_tsv(params=model_params,
                        feature_name=feature_names[feature_index])
