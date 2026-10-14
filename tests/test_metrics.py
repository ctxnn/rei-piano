"""Our fast F1 must equal RoboPianist's scikit-learn version, number for number."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from runaway.env import per_step_precision_recall_f1


def _sklearn_reference(y_true, y_pred):
    rows = [
        precision_recall_fscore_support(t, p, average="binary", zero_division=1)[:3]
        for t, p in zip(y_true.astype(float), y_pred.astype(float))
    ]
    return np.mean(rows, axis=0)


def test_matches_sklearn_on_random_and_edge_cases():
    rng = np.random.default_rng(0)
    y_true = rng.random((500, 88)) < 0.02
    y_pred = rng.random((500, 88)) < 0.02
    y_true[:50] = False  # silence, correctly silent
    y_pred[:25] = False
    y_pred[50:60] = True  # many wrong keys
    y_true[60:70] = True
    y_pred[60:70] = False  # all keys missed
    ours = per_step_precision_recall_f1(y_true, y_pred)
    ref = _sklearn_reference(y_true, y_pred)
    np.testing.assert_allclose([ours.precision, ours.recall, ours.f1], ref, rtol=0, atol=1e-12)
