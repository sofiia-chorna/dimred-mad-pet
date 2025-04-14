import numpy as np
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import explained_variance_score, mean_absolute_error


def compute_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    cos_sim = 1 - cosine(actual.flatten(), predicted.flatten())
    var_score = explained_variance_score(actual, predicted)
    return {"mae": mae, "cosine similarity": cos_sim, "var_score": var_score}


def eval(pred_dict, ground_truth_dict):
    all_actual = np.concatenate(list(ground_truth_dict.values()))
    all_predicted = np.concatenate(
        [v.numpy() if isinstance(v, torch.Tensor) else v for v in pred_dict.values()]
    )
    return compute_metrics(all_actual, all_predicted)
