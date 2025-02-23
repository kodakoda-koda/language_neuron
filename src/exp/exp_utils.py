import os
from typing import Dict

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

    precision = cumulative_TP / (cumulative_TP + cumulative_FP)

    return np.sum(precision * y_true_sorted, axis=0) / np.sum(y_true)


def compute_ap(neurons: np.ndarray, labels: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
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


def plot_indices(indices: Dict[str, Dict[str, np.ndarray]], num_layers: int, plot_path: str) -> None:
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        top = indices[l]["top"]
        middle = indices[l]["middle"]
        bottom = indices[l]["bottom"]

        plt.figure(figsize=(15, 5))
        for j, idx in enumerate([top, middle, bottom]):
            plt.subplot(1, 3, j + 1)
            plt.hist(idx, bins=num_layers)
            plt.xticks(np.arange(num_layers), np.arange(1, num_layers + 1))
            plt.title(["top", "middle", "bottom"][j])
        plt.suptitle(l)
        plt.savefig(plot_path + f"/{l}.png")
        plt.close()

    corr = np.zeros((len(lang), len(lang)))
    for i, l1 in enumerate(lang):
        for j, l2 in enumerate(lang):
            l1_top_bottom = set(indices[l1]["top"]).union(set(indices[l1]["bottom"]))
            l2_top_bottom = set(indices[l2]["top"]).union(set(indices[l2]["bottom"]))
            corr[i, j] = len(l1_top_bottom.intersection(l2_top_bottom))

    sns.heatmap(corr, annot=True)
    plt.xticks(np.arange(len(lang)), lang)
    plt.yticks(np.arange(len(lang)), lang)
    plt.savefig(plot_path + "/heatmap.png")
    plt.close()


def intervention_indices(num_layers, d_model, top_bottom_indices):
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
