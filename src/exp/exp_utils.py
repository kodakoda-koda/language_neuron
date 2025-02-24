import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def min_max_scaler(data: np.ndarray) -> np.ndarray:
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    return (data - min_) / (max_ - min_ + 1e-10)


def average_precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_sorted = y_true[np.argsort(y_pred, axis=0)[::-1]]

    cumulative_TP = np.cumsum(y_true_sorted, axis=0)
    cumulative_FP = np.cumsum(1 - y_true_sorted, axis=0)

    precision = cumulative_TP / (cumulative_TP + cumulative_FP + 1e-10)

    return np.sum(precision * y_true_sorted, axis=0) / (np.sum(y_true) + 1e-10)


def compute_ap(neurons: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, list]]:
    neurons = min_max_scaler(neurons)
    indices = {}
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        ap = average_precision(labels[:, i], neurons)
        top_1000 = np.argsort(ap)[::-1][:1000]
        middle_1000 = np.argsort(ap)[::-1][len(ap) // 2 - 500 : len(ap) // 2 + 500]
        bottom_1000 = np.argsort(ap)[:1000]
        indices[l] = {
            "top": top_1000.tolist(),
            "middle": middle_1000.tolist(),
            "bottom": bottom_1000.tolist(),
        }
    return indices


def intervention_indices(num_layers: int, d_model: int, top_bottom_indices: List[int]):
    neuron_indices = []
    hidden_indices = []

    range_ = np.arange(num_layers * d_model * 9).reshape(num_layers, d_model * 9)
    q_indices = range_[:, :d_model]
    k_indices = range_[:, d_model : 2 * d_model]
    v_indices = range_[:, 2 * d_model : 3 * d_model]
    o1_indices = range_[:, 3 * d_model : 4 * d_model]
    f_indices = range_[:, 4 * d_model : 8 * d_model]
    o2_indices = range_[:, 8 * d_model : 9 * d_model]

    for i in range(num_layers):
        neuron_indices_ = [[] for _ in range(6)]
        hidden_indices_ = [[] for _ in range(6)]

        for j, k in enumerate(top_bottom_indices):
            if k in q_indices[i]:
                neuron_indices_[0].append(j)
                hidden_indices_[0].append(k % (d_model * 9))
            elif k in k_indices[i]:
                neuron_indices_[1].append(j)
                hidden_indices_[1].append(k % (d_model * 9) - d_model)
            elif k in v_indices[i]:
                neuron_indices_[2].append(j)
                hidden_indices_[2].append(k % (d_model * 9) - 2 * d_model)
            elif k in o1_indices[i]:
                neuron_indices_[3].append(j)
                hidden_indices_[3].append(k % (d_model * 9) - 3 * d_model)
            elif k in f_indices[i]:
                neuron_indices_[4].append(j)
                hidden_indices_[4].append(k % (d_model * 9) - 4 * d_model)
            elif k in o2_indices[i]:
                neuron_indices_[5].append(j)
                hidden_indices_[5].append(k % (d_model * 9) - 8 * d_model)

        neuron_indices.append(neuron_indices_)
        hidden_indices.append(hidden_indices_)

    return neuron_indices, hidden_indices
