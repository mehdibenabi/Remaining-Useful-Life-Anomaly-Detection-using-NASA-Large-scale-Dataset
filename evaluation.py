import torch
import numpy as np


def nasa_score_tensor(a1 , a2, y_hat, y): 
    """
    y_hat, y: tensors of shape [batch_size]
    returns: scalar total score over batch
    """
    d = y_hat - y  # [batch]

    # boolean masks
    neg_mask = d < 0
    pos_mask = ~neg_mask

    score = torch.zeros_like(d)

    # under-prediction
    score[neg_mask] = torch.exp(-d[neg_mask] / a1) - 1.0
    # over-prediction
    score[pos_mask] = torch.exp(d[pos_mask] / a2) - 1.0

    return score.sum()  # total batch score 

#========================================================================================================================================

def masked_mse(pred, target, mask):
    return ((pred - target) ** 2)[mask].mean()

def masked_mae(pred, target, mask):
    return (pred - target).abs()[mask].mean()

#========================================================================================================================================

def r2_score_tensor(preds, targets):
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - targets.mean()) ** 2) + 1e-12
    return (1 - ss_res / ss_tot).item()

#========================================================================================================================================

def expected_calibration_error(probs, targets, n_bins=15):
    """
    Binary ECE: bins samples by confidence in the predicted class
    (max(p, 1-p)) and compares average confidence to average accuracy
    within each bin.
    """
    probs = np.asarray(probs).reshape(-1)
    targets = np.asarray(targets).reshape(-1)
    preds = (probs >= 0.5).astype(np.float32)
    confidences = np.where(preds == 1, probs, 1 - probs)
    correct = (preds == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (count / n) * abs(bin_acc - bin_conf)

    return float(ece)
