# metrics.py
import math

import torch


def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item() * 100


def expected_calibration_error(logits, y, n_bins=15):
    # Bin predictions by confidence; ECE = Σ (bin share) · |accuracy in bin − mean confidence in bin|.
    # 0 means "when the model says 90%, it is right 90% of the time".
    probs = logits.float().softmax(1)
    conf, pred = probs.max(1)
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=logits.device)
    ece = torch.zeros((), device=logits.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.float().mean() * (correct[m].mean() - conf[m].mean()).abs()
    return ece.item() * 100


def confusion_matrix(preds, y, n):
    cm = torch.zeros(n, n, dtype=torch.long)
    cm.index_put_((y.cpu(), preds.cpu()), torch.ones_like(y.cpu()), accumulate=True)
    return cm


def mcnemar_p(correct_a, correct_b):
    # Exact two-sided McNemar test on paired predictions (same test clips): only disagreements count
    n10 = int((correct_a & ~correct_b).sum())
    n01 = int((~correct_a & correct_b).sum())
    n, k = n10 + n01, min(n10, n01)
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
