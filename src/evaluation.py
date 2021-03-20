import numpy as np

try:
    from editdistance import eval as lsdistance
except ImportError:
    from Levenshtein import distance as lsdistance


def eval_ld_batch(y_true: list[str], y_pred: list[str]):
    lds = [lsdistance(y, y_) for y, y_ in zip(y_true, y_pred)]
    mean_ld = np.mean(lds)
    return mean_ld
