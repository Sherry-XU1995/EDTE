import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score


class Metric:
    def __init__(self, args):
        self.scores = []
        # self.threshold = args.threshold
        self.threshold = -1

    def add_result(self, result):
        y, y_prob = result
        self.scores.append(self._get_scores(y, y_prob))

    def _get_scores(self, y, y_prob):
        y = np.array(y)
        y_prob = np.array(y_prob)
        min_prob = np.nanmin(y_prob)
        mean_prob = np.nanmean(y_prob)
        median_prob = np.nanmedian(y_prob)
        max_prob = np.nanmax(y_prob)
        nan_index = np.isnan(y_prob)
        y_prob[nan_index] = 0
        y[nan_index] = 1
        threshold = self.threshold if self.threshold != -1 else np.nanmedian(y_prob)
        y_pred = y_prob > threshold
        if np.unique(y).size == 1:
            auc = 0
        else:
            auc = roc_auc_score(y, y_prob)
        return {
            'min_prob': min_prob,
            'mean_prob': mean_prob,
            'median_prob': median_prob,
            'max_prob': max_prob,
            'acc': accuracy_score(y, y_pred),
            'p': precision_score(y, y_pred),
            'r': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc': auc,
            'ap': average_precision_score(y, y_prob),
        }

    def get_scores(self):
        final_scores = {}
        for k in self.scores[0].keys():
            final_scores[k] = np.mean([_[k] for _ in self.scores])
        return final_scores

# class Metric:
#     def __init__(self, threshold=0.5):
#         self.y = []
#         self.y_prob = []
#         self.threshold = threshold
#
#     def init(self):
#         self.y = []
#         self.y_prob = []
#
#     def add_result(self, result):
#         y, y_prob = result
#         self.y += y.reshape(-1).tolist()
#         self.y_prob += y_prob.reshape(-1).tolist()
#
#     def get_scores(self):
#         y = np.array(self.y)
#         y_prob = np.array(self.y_prob)
#         y_pred = y_prob > self.threshold
#         return {
#             'acc': accuracy_score(y, y_pred),
#             'p': precision_score(y, y_pred),
#             'r': recall_score(y, y_pred),
#             'f1': f1_score(y, y_pared),
#             'auc': roc_auc_score(y, y_prob),
#             'ap': average_precision_score(y, y_prob),
#         }
