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
    scores = {}
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        ap = average_precision(labels[:, i], neurons)
        top_1000 = np.argsort(ap)[::-1][:1000]
        middle_1000 = np.argsort(ap)[::-1][len(ap) // 2 - 500 : len(ap) // 2 + 500]
        bottom_1000 = np.argsort(ap)[:1000]
        scores[l] = {
            "ap": ap,
            "top": top_1000,
            "middle": middle_1000,
            "bottom": bottom_1000,
        }
    return scores


def plot_scores(scores: Dict[str, Dict[str, np.ndarray]], num_layers: int) -> None:
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        top = scores[l]["top"]
        middle = scores[l]["middle"]
        bottom = scores[l]["bottom"]

        plt.figure(figsize=(15, 5))
        for j, idx in enumerate([top, middle, bottom]):
            plt.subplot(1, 3, j + 1)
            plt.hist(idx, bins=num_layers)
            plt.xticks(np.arange(num_layers), np.arange(1, num_layers + 1))
            plt.title(["top", "middle", "bottom"][j])
        plt.suptitle(l)
        plt.savefig(f"./tmp/{l}.png")
        plt.close()

    corr = np.zeros((len(lang), len(lang)))
    for i, l1 in enumerate(lang):
        for j, l2 in enumerate(lang):
            l1_top_bottom = set(scores[l1]["top"]).union(set(scores[l1]["bottom"]))
            l2_top_bottom = set(scores[l2]["top"]).union(set(scores[l2]["bottom"]))
            corr[i, j] = len(l1_top_bottom.intersection(l2_top_bottom))

    sns.heatmap(corr, annot=True)
    plt.xticks(np.arange(len(lang)), lang)
    plt.yticks(np.arange(len(lang)), lang)
    plt.savefig("./tmp/corr.png")
    plt.close()
