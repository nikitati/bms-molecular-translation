import numpy as np
import editdistance


def eval_ld_batch(y_true: list[str], y_pred: list[str]):
    lds = [editdistance.eval(y, y_) for y, y_ in zip(y_true, y_pred)]
    mean_ld = np.mean(lds)
    return mean_ld

