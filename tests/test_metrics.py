import numpy as np
from src.metrics import pair_scores, choose_threshold_at_far, evaluate_with_threshold


def test_pair_scores_has_same_and_different_pairs():
    emb = np.array([[1., 0.], [0.99, 0.01], [0., 1.], [0.01, 0.99]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    scores, same = pair_scores(emb, labels)
    assert len(scores) == 6
    assert same.sum() == 2
    assert (~same).sum() == 4


def test_evaluate_metrics_return_rates():
    emb = np.array([[1., 0.], [0.99, 0.01], [0., 1.], [0.01, 0.99]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    result = evaluate_with_threshold(emb, labels, threshold=0.8)
    assert result["far"] == 0.0
    assert result["frr"] == 0.0
    assert result["f1"] == 1.0
