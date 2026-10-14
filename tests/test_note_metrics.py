"""Note-level metrics: on-time strikes count, chatter and brushed neighbours are errors."""

import numpy as np

from runaway.evaluate import note_metrics


def _roll(T, notes):
    m = np.zeros((T, 88), bool)
    for key, start, end in notes:
        m[start:end, key] = True
    return m


def test_perfect_and_late_and_chatter():
    goal = _roll(100, [(60, 10, 30), (60, 40, 60), (62, 70, 90)])
    assert note_metrics(goal, goal)["note_f1"] == 1.0
    # One strike 3 steps late (> 100 ms): one miss, one extra onset.
    late = _roll(100, [(60, 10, 30), (60, 43, 60), (62, 70, 90)])
    m = note_metrics(goal, late)
    assert m["note_recall"] == 2 / 3 and m["note_precision"] == 2 / 3
    # Chatter: the key bounces up and down inside a held note -> extra onsets.
    chatter = goal.copy()
    chatter[20, 60] = False
    chatter[25, 60] = False
    m = note_metrics(goal, chatter)
    assert m["note_recall"] == 1.0 and m["presses"] == 5 and abs(m["note_precision"] - 3 / 5) < 1e-9
