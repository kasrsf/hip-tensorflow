import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import operator

from Multi_THIP import Multi_TensorHIP


def compute_error(cost_vector):
    return np.sqrt(cost_vector.sum() / len(cost_vector))

def group_tags_by_date(df, hashtag, all_dates, print_plot=False):
    df_tags = df[df.hashtag == hashtag]
    tags_grouped = df_tags.groupby([df_tags['create_time'].dt.date]).size()
    tags_grouped = tags_grouped.reset_index()
    tags_grouped.columns = ['Time', 'Frequency']
    vals = all_dates.set_index('Time').join(tags_grouped.set_index('Time')).fillna(0).astype('int').values.flatten()

    if print_plot == True:
        plt.plot(vals)
        plt.title("#" + hashtag + " Occurances by Day")
        plt.show()

    return vals

def plot_predictions(truth_value, adam_pred, adagrad_pred, hip_pred, num_train, num_test, title=""):
    last_index = num_train + num_test
    plt.axvline(num_train, color='k')

    plt.plot(np.arange(last_index), truth_value[:last_index], 'k--', label='observed #views')

    if len(adam_pred) > 0:
        plt.plot(np.arange(last_index), adam_pred[:last_index], 'b-', label='ADAM fit')
    if len(adagrad_pred) > 0:
        plt.plot(np.arange(last_index), adagrad_pred[:last_index], 'y-', label='ADAGRAD fit')
    if len(hip_pred) > 0:
        plt.plot(np.arange(last_index), hip_pred[:last_index], 'g-', label='HIP fit')

    plt.legend([plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
                        plt.Line2D((0, 1), (0, 0), color='g'),
                        plt.Line2D((0, 1), (0, 0), color='b'),
                        plt.Line2D((0, 1), (0, 0), color='y')],
                       ['Observed view', 'Original HIP Fit', 'TF ADAM', 'TF ADAGRAD'],
                       frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.125),
                       fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Day')

    plt.title(title)

    plt.show()

def plot_improved_predictions(truth_value, single_pred, multi_pred, hip_pred, num_train, num_test, title=""):
    last_index = num_train + num_test
    plt.axvline(num_train, color='k')

    plt.plot(np.arange(last_index), truth_value[:last_index], 'k--', label='observed #views')

    if len(single_pred) > 0:
        plt.plot(np.arange(last_index), single_pred[:last_index], 'b-', label='Single TensorHIP fit')
    if len(multi_pred) > 0:
        plt.plot(np.arange(last_index), multi_pred[:last_index], 'y-', label='Multi TensorHIP fit')
    if len(hip_pred) > 0:
        plt.plot(np.arange(last_index), hip_pred[:last_index], 'g-', label='HIP fit')

    plt.legend([plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
                        plt.Line2D((0, 1), (0, 0), color='g'),
                        plt.Line2D((0, 1), (0, 0), color='b'),
                        plt.Line2D((0, 1), (0, 0), color='y')],
                       ['Observed view', 'Original HIP Fit', 'Single TensorHIP fit', 'Multi TensorHIP fit'],
                       frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.125),
                       fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Day')

    plt.title(title)

    plt.show()    
    
def plot_multistream_predictions(truth_value, top_weighted_pred, top_mi_pred, num_train, num_test, title=""):
    last_index = num_train + num_test
    plt.axvline(num_train, color='k')

    plt.plot(np.arange(last_index), truth_value[:last_index], 'k--', label='observed #views')
    
    plt.plot(np.arange(last_index), top_weighted_pred[:last_index], 'g-', label='TopK Weighted Predictors')
    plt.plot(np.arange(last_index), top_mi_pred[:last_index], 'b-', label='Topk Mutual Information Scores Predictors')
    
    plt.legend([plt.Line2D((0, 1), (0, 0), color='k', linestyle='--'),
                        plt.Line2D((0, 1), (0, 0), color='g'),
                        plt.Line2D((0, 1), (0, 0), color='b')],
                       ['Observed view', 'TopK Weights', 'TopK MI'],
                       frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.125),
                       fancybox=True, shadow=True, ncol=4)

    plt.xlabel('Day')

    plt.title(title)

    plt.show()

def get_hashtag_predictors(freq_dict, target):
    predictors = dict()

    for key, value in freq_dict.items():
        if key not in [target]:
            predictors[key] = value

    return list(predictors.keys()), list(predictors.values())

def get_mutual_info_scores(freq_dict, hashtag, sort=True):
    mi_scores = dict()

    for key, value in freq_dict.items():
        if key not in [hashtag]:
            mi_scores[key] = mutual_info_score(freq_dict[hashtag], value) 

    if sort == True:
        mi_scores = sorted(mi_scores.items(), key=operator.itemgetter(1), reverse=True)
    
    return mi_scores

def get_topk_mi_scores(freq_dict, hashtag, k=20):
    sorted_scores = get_mutual_info_scores(freq_dict, hashtag)
    keys = [score[0] for score in sorted_scores[:k]]
    values = [freq_dict[key] for key in keys]

    return keys, values


def run_experiment(freq_dict, hashtag, num_train, num_test, k=20, num_initializations=5, print_plot=True, verbose=True):
    topk_mi_keys, topk_mi_vals = get_topk_mi_scores(freq_dict, hashtag, k)
    topk_mi_timeseries = [freq_dict[key] for key in topk_mi_keys]

    predictor_keys, predictor_values = get_hashtag_predictors(freq_dict, hashtag)

    if verbose == True:
        print("** Optimizing model with all hashtags")
    # get topk weighted in optimized model
    hip = Multi_TensorHIP(predictor_values, freq_dict[hashtag], num_train, num_test)
    hip.fit(num_initializations, op='adagrad', verbose=verbose)

    top_weighted = list(np.array(predictor_keys)[np.absolute(np.array(hip.get_mu()[0])).argsort()[-k:][::-1]])
    top_weighted_timeseries = [freq_dict[key] for key in top_weighted]
    
    if verbose == True:    
        print("** Optimizing model with topk MI hashtags")
    mi_hip = Multi_TensorHIP(topk_mi_timeseries, freq_dict[hashtag], num_train, num_test)
    mi_preds, mi_losses = mi_hip.fit(num_initializations, op='adagrad', verbose=verbose)

    if verbose == True:
        print("** Optimizing model with topk weighted hashtags")
    weight_hip = Multi_TensorHIP(top_weighted_timeseries, freq_dict[hashtag], num_train, num_test)
    weight_preds, weight_losses = weight_hip.fit(num_initializations, op='adagrad', verbose=verbose)

    mi_error = compute_error(mi_losses)
    weight_error = compute_error(weight_losses)
    
    if verbose == True:    
        print("MI Error = " + str(mi_error))
        print("Top Weighted Error = " + str(weight_error))

    if print_plot == True:
        plot_multistream_predictions(freq_dict[hashtag], weight_preds, mi_preds, num_train, num_test, \
             title="Models for #" + hashtag)    

    return topk_mi_keys, top_weighted, mi_error, weight_error
