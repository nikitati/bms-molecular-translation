from typing import List

import numpy as np

try:
    from editdistance import eval as lsdistance
except ImportError:
    from Levenshtein import distance as lsdistance


def eval_ld_batch(y_true: List[str], y_pred: List[str]):
    lds = [lsdistance(y, y_) for y, y_ in zip(y_true, y_pred)]
    mean_ld = np.mean(lds)
    return mean_ld


class Evaluator:

    def __init__(self, dataset):
        self.dataset = dataset

    def eval_batch(self, idxs, y_pred):
        original_targets = self.dataset.get_original_targets(idxs)
        ld = eval_ld_batch(original_targets, y_pred)
        return ld
