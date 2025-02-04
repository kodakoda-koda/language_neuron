from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def min_max_scaler(data: np.ndarray) -> np.ndarray:
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    return (data - min_) / (max_ - min_ + 1e-10)


def average_precision_(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true_sorted = y_true[np.argsort(y_pred, axis=0)[::-1]]

    cumulative_TP = np.cumsum(y_true_sorted, axis=0)
    cumulative_FP = np.cumsum(1 - y_true_sorted, axis=0)

    n_positive = np.sum(y_true)
    precision = cumulative_TP / (cumulative_TP + cumulative_FP)
    recall = cumulative_TP / n_positive

    precision = np.concatenate([np.zeros((y_true.shape[0], 1)), precision, np.zeros((y_true.shape[0], 1))], axis=1)
    recall = np.concatenate([np.zeros((y_true.shape[0], 1)), recall, np.ones((y_true.shape[0], 1))], axis=1)
    return np.sum((recall[1:] - recall[:-1]) * precision[:-1], axis=0)


def compute_score(neurons: np.ndarray, labels: np.ndarray) -> Dict[str, np.ndarray]:
    neurons = min_max_scaler(neurons)
    scores = {}
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        scores[l] = average_precision_(labels[:, i], neurons)
    return scores


def plot_scores(scores: Dict[str, np.ndarray]) -> None:
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        top_1000 = np.argsort(scores[l])[::-1][:1000]
        middle_1000 = np.argsort(scores[l])[::-1][len(scores[l]) // 2 - 500 : len(scores[l]) // 2 + 500]
        bottom_1000 = np.argsort(scores[l])[:1000]

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(top_1000, bins=24)
        plt.title(f"{l} top 1000 scores")
        plt.subplot(1, 3, 2)
        plt.hist(middle_1000, bins=24)
        plt.title(f"{l} middle 1000 scores")
        plt.subplot(1, 3, 3)
        plt.hist(bottom_1000, bins=24)
        plt.title(f"{l} bottom 1000 scores")
        plt.savefig(f"./tmp/{l}.png")
        plt.close()
