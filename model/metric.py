import numpy as np

def cat_metric(y_pred, y_true):
    assert len(y_pred) == len(y_true)
    correct = 0
    for pred, true in zip(y_pred, y_true):
        if np.array_equal(pred, true):
            correct += 1

    return correct / len(y_pred)
