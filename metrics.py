import math
import numpy as np
import torch

from collections import defaultdict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
# from pdb import set_trace
# from utils.general_util import log


class Meter(object):

    def __init__(self, metrics, main_metric):
        self.metrics = metrics
        self.main_metric = main_metric
        self.initial_metric = 10E9 \
            if self.metrics[main_metric].better == "low" \
            else -10E9
        self.reset()

    def reset(self):
        self.predictions = []
        self.targets = []
        self.results = {}

    def is_better(self, results, value):
        if self.metrics[self.main_metric].better == "low":
            return results[self.main_metric] < value
        else:
            return results[self.main_metric] > value

    def update(self, predictions, targets):
        predictions = predictions.cpu().float().detach().numpy()
        targets = targets.cpu().float().detach().numpy()

        self.predictions.append(predictions)
        self.targets.append(targets)

        results = {}
        for metric_name, metric_fn in self.metrics.items():
            results[metric_name] = metric_fn.calculate(predictions, targets)
        return results

    def compute(self):
        for metric_name, metric_fn in self.metrics.items():
            self.results[metric_name] = metric_fn.calculate(
                np.concatenate(self.predictions),
                np.concatenate(self.targets)
            )
        return self.results


class Metric(object):
    def __init__(self, better="low", kwargs={}):
        self.better = better
        self.kwargs = kwargs

    def calculate(self, predictions, targets):
        pass


class R2(Metric):
    def __init__(self, kwargs={}):
        super(R2, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        return r2_score(y_true=targets.reshape(-1), y_pred=predictions.reshape(-1), **self.kwargs)

class MAE(Metric):
    def calculate(self, predictions, targets):
        return mean_absolute_error(y_true=targets.reshape(-1), y_pred=predictions.reshape(-1), **self.kwargs)

class MSE(Metric):
    def calculate(self, predictions, targets):
        return mean_squared_error(y_true=targets.reshape(-1), y_pred=predictions.reshape(-1), **self.kwargs)

class RMSE(Metric):
    def calculate(self, predictions, targets):
        return math.sqrt(mean_squared_error(y_true=targets.reshape(-1), y_pred=predictions.reshape(-1), **self.kwargs))

class PearsonR(Metric):
    def __init__(self, kwargs={}):
        super(PearsonR, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        try:
            return stats.pearsonr(x=targets.reshape(-1), y=predictions.reshape(-1), **self.kwargs)[0]
        except:
            set_trace()

class SpearmanR(Metric):
    def __init__(self, kwargs={}):
        super(SpearmanR, self).__init__("high", **kwargs)

    def calculate(self, predictions, targets):
        return stats.spearmanr(targets.reshape(-1), predictions.reshape(-1), **self.kwargs)[0]
