import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ========================================================================================================================================
# Regression (RUL) plots
# ========================================================================================================================================

def plot_predictions(targets, preds, split_name="Test"):
    targets = _to_numpy(targets)
    preds = _to_numpy(preds)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(targets, label="Target", linewidth=2)
    plt.plot(preds, label="Predicted", linewidth=2)
    plt.title(f"{split_name} - Target vs Predicted")
    plt.xlabel("Sample")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(targets, preds, alpha=0.5)
    mn = min(targets.min(), preds.min())
    mx = max(targets.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    plt.title(f"{split_name} - Real vs Predicted")
    plt.xlabel("Target RUL")
    plt.ylabel("Predicted RUL")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_residuals(targets, preds, split_name="Test"):
    targets = _to_numpy(targets)
    preds = _to_numpy(preds)
    residuals = preds - targets

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(residuals, linewidth=1)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title(f"{split_name} - Residuals (Predicted - Target)")
    plt.xlabel("Sample")
    plt.ylabel("Residual")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    over = residuals[residuals >= 0]
    under = residuals[residuals < 0]
    plt.hist(under, bins=30, alpha=0.6, color='tab:blue', label=f"Under-prediction (n={len(under)})")
    plt.hist(over, bins=30, alpha=0.6, color='tab:red', label=f"Over-prediction (n={len(over)})")
    plt.axvline(0, color='black', linestyle='--', linewidth=1)
    plt.title(f"{split_name} - Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_uncertainty(targets, preds, std, split_name="Test"):

    targets = _to_numpy(targets)
    preds = _to_numpy(preds)
    std = _to_numpy(std)
    x = np.arange(len(targets))

    plt.figure(figsize=(12, 5))
    plt.plot(x, targets, label="True", linewidth=2)
    plt.plot(x, preds, label="Predicted", linewidth=2)
    plt.fill_between(x, preds - 2 * std, preds + 2 * std, alpha=0.3, label="95% Confidence")
    plt.legend()
    plt.xlabel("Sample")
    plt.ylabel("RUL")
    plt.title(f"{split_name} - Prediction with Uncertainty")
    plt.grid(True)
    plt.show()


# ========================================================================================================================================
# Training-curve plots (loss / rmse / r2 / mae )
# ========================================================================================================================================

def plot_metric_curve(train_values, val_values=None, metric_name="Loss", title=None):

    epochs = np.arange(1, len(train_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label=f"Train {metric_name}", linewidth=2)
    if val_values is not None:
        plt.plot(epochs, val_values, label=f"Val {metric_name}", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(title or f"{metric_name} over Training")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_history(history, metric, title=None):

    train_key, val_key = f"train_{metric}", f"val_{metric}"
    train_values = [h[train_key] for h in history]
    val_values = [h[val_key] for h in history] if val_key in history[0] else None
    plot_metric_curve(train_values, val_values, metric_name=metric.upper(), title=title)


# ========================================================================================================================================
# Classification plots
# ========================================================================================================================================

def plot_confusion_matrix(targets, probs, threshold=0.5, class_names=("Negative", "Positive"), split_name="Test"):
    targets = _to_numpy(targets)
    probs = _to_numpy(probs)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"{split_name} - Confusion Matrix")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, f"{labels[i, j]}\n{cm[i, j]}", ha="center", va="center", color=color)

    plt.colorbar()
    plt.tight_layout()
    plt.show()


def plot_roc_curve(targets, probs, split_name="Test"):
    targets = _to_numpy(targets)
    probs = _to_numpy(probs)

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} - ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_calibration_curve(targets, probs, n_bins=15, split_name="Test"):

    targets = _to_numpy(targets)
    probs = _to_numpy(probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        bin_centers.append(probs[in_bin].mean())
        bin_accs.append(targets[in_bin].mean())
        bin_counts.append(in_bin.sum())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label="Perfectly calibrated")
    plt.plot(bin_centers, bin_accs, marker='o', linewidth=2, label="Model")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title(f"{split_name} - Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_classification_dashboard(targets, probs, threshold=0.5, split_name="Test"):

    targets = _to_numpy(targets)
    probs = _to_numpy(probs)
    preds = (probs >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    cm = confusion_matrix(targets, preds, labels=[0, 1])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title(f"{split_name} - Confusion Matrix")
    axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh else "black"
            axes[0].text(j, i, f"{labels[i, j]}\n{cm[i, j]}", ha="center", va="center", color=color)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    fpr, tpr, _ = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
    axes[1].plot([0, 1], [0, 1], 'r--', linewidth=1)
    axes[1].set_title(f"{split_name} - ROC Curve")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(); axes[1].grid(True)

    bin_edges = np.linspace(0.0, 1.0, 16)
    bin_centers, bin_accs = [], []
    for i in range(15):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (probs >= lo) & (probs <= hi) if i == 0 else (probs > lo) & (probs <= hi)
        if in_bin.sum() == 0:
            continue
        bin_centers.append(probs[in_bin].mean())
        bin_accs.append(targets[in_bin].mean())
    axes[2].plot([0, 1], [0, 1], 'r--', linewidth=1, label="Perfectly calibrated")
    axes[2].plot(bin_centers, bin_accs, marker='o', linewidth=2, label="Model")
    axes[2].set_title(f"{split_name} - Calibration")
    axes[2].set_xlabel("Mean Predicted Probability"); axes[2].set_ylabel("Observed Frequency")
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def plot_regression_dashboard(targets, preds, std=None, split_name="Test"):

    targets = _to_numpy(targets)
    preds = _to_numpy(preds)
    residuals = preds - targets
    n_panels = 4 if std is not None else 3

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))

    axes[0].plot(targets, label="Target", linewidth=2)
    axes[0].plot(preds, label="Predicted", linewidth=2)
    axes[0].set_title(f"{split_name} - Target vs Predicted")
    axes[0].set_xlabel("Sample"); axes[0].set_ylabel("RUL")
    axes[0].legend(); axes[0].grid(True)

    mn, mx = min(targets.min(), preds.min()), max(targets.max(), preds.max())
    axes[1].scatter(targets, preds, alpha=0.5)
    axes[1].plot([mn, mx], [mn, mx], 'r--', linewidth=2)
    axes[1].set_title(f"{split_name} - Real vs Predicted")
    axes[1].set_xlabel("Target RUL"); axes[1].set_ylabel("Predicted RUL")
    axes[1].grid(True)

    over, under = residuals[residuals >= 0], residuals[residuals < 0]
    axes[2].hist(under, bins=30, alpha=0.6, color='tab:blue', label=f"Under (n={len(under)})")
    axes[2].hist(over, bins=30, alpha=0.6, color='tab:red', label=f"Over (n={len(over)})")
    axes[2].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_title(f"{split_name} - Residual Distribution")
    axes[2].set_xlabel("Residual"); axes[2].set_ylabel("Count")
    axes[2].legend(); axes[2].grid(True)

    if std is not None:
        std = _to_numpy(std)
        x = np.arange(len(targets))
        axes[3].plot(x, targets, label="True", linewidth=2)
        axes[3].plot(x, preds, label="Predicted", linewidth=2)
        axes[3].fill_between(x, preds - 2 * std, preds + 2 * std, alpha=0.3, label="95% Confidence")
        axes[3].set_title(f"{split_name} - Uncertainty")
        axes[3].set_xlabel("Sample"); axes[3].set_ylabel("RUL")
        axes[3].legend(); axes[3].grid(True)

    plt.tight_layout()
    plt.show()


# ========================================================================================================================================
# SSL backbone (masked reconstruction) plots
# ========================================================================================================================================

def plot_ssl_history(history):

    epochs = np.arange(1, len(history) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train", linewidth=2)
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="Val", linewidth=2)
    axes[0].set_title("SSL Masked-MSE Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, [h["train_mae"] for h in history], label="Train", linewidth=2)
    axes[1].plot(epochs, [h["val_mae"] for h in history], label="Val", linewidth=2)
    axes[1].set_title("SSL Masked MAE")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_ssl_reconstruction(x, pred, mask, feature_idx=0, split_name="Val"):

    x = _to_numpy(x)[:, feature_idx]
    pred = _to_numpy(pred)[:, feature_idx]
    mask = _to_numpy(mask)[:, feature_idx].astype(bool)
    t = np.arange(len(x))

    plt.figure(figsize=(12, 5))
    plt.plot(t, x, label="True", linewidth=2, color='tab:blue')
    plt.scatter(t[mask], x[mask], color='black', marker='x', s=60, label="Masked (true)", zorder=5)
    plt.scatter(t[mask], pred[mask], color='tab:red', marker='o', s=40, label="Reconstructed", zorder=5)
    plt.title(f"{split_name} - SSL Reconstruction (feature {feature_idx})")
    plt.xlabel("Timestep")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_ssl_feature_errors(x, pred, mask, feature_names=None, split_name="Val"):

    x = _to_numpy(x)
    pred = _to_numpy(pred)
    mask = _to_numpy(mask).astype(bool)

    num_features = x.shape[-1]
    errors = np.full(num_features, np.nan)
    for f in range(num_features):
        feat_mask = mask[..., f]
        if feat_mask.sum() == 0:
            continue
        errors[f] = np.abs(pred[..., f][feat_mask] - x[..., f][feat_mask]).mean()

    labels = feature_names if feature_names is not None else [str(i) for i in range(num_features)]

    plt.figure(figsize=(max(8, num_features * 0.4), 5))
    plt.bar(labels, errors)
    plt.xticks(rotation=90)
    plt.ylabel("Masked MAE")
    plt.title(f"{split_name} - SSL Reconstruction Error per Feature")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()
