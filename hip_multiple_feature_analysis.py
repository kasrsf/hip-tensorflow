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
    
    if len(sys.argv) == 2:
        input_path = sys.argv[1]
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
            features, target, feature_names, target_name = load_data_from_csv(file_path)
            xs.append(features)
            input_feature_names = feature_names
            ys.append(target)
            file_paths.append(file_path)
    
    sys.stderr.write("beginning the training\n")
    sys.stderr.flush()
    start_time = time.time()
    # eta default random initialization
    hip_model = TensorHIP(xs=xs, ys=ys,    
                            feature_names=input_feature_names,
                            l1_param=0, l2_param=0,
                            verbose=False)
    hip_model.train()    
    sys.stderr.write("\ntraining completed in {} seconds\n".format(time.time() - start_time))
    sys.stderr.flush()
    
    print(hip_model.get_params_df().to_csv(sep='\t', index=False))