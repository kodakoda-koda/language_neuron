import matplotlib.pyplot as plt
import numpy as np


def min_max_scaler(data: np.ndarray) -> np.ndarray:
    min_ = np.min(data, axis=0)
    max_ = np.max(data, axis=0)
    return (data - min_) / (max_ - min_ + 1e-10)


def average_precision(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int32)
    y_pred = y_pred.astype(np.float32)
    y_true_sorted = y_true[np.argsort(y_pred)[::-1]]
    # n = len(y_true)
    n_positive = np.sum(y_true)
    # n_negative = n - n_positive
    cumulative_TP = np.cumsum(y_true_sorted)
    cumulative_FP = np.cumsum(1 - y_true_sorted)
    precision = cumulative_TP / (cumulative_TP + cumulative_FP)
    recall = cumulative_TP / n_positive
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])
    return np.sum((recall[1:] - recall[:-1]) * precision[:-1])


def compute_score(neurons, labels):
    neurons = min_max_scaler(neurons)
    scores = {}
    lang = ["en", "de", "fr", "es", "zh", "ja"]
    for i, l in enumerate(lang):
        scores_l = []
        for j in range(neurons.shape[1]):
            scores_l.append(average_precision(labels[:, i], neurons[:, j]))
        scores[l] = scores_l
    return scores


def plot_scores(scores):
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
