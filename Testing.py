import time
import torch
import gpytorch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from evaluation import nasa_score_tensor, masked_mse, masked_mae, r2_score_tensor, expected_calibration_error

def test(model, test_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    a1 = 13.0
    a2 = 10.0

    test_loss = 0.0
    test_preds = []
    test_targets = []
    test_nasa_total = 0.0
    start_time = time.time()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).float().squeeze(-1)

            preds = model(X_batch).squeeze(-1)
            loss = criterion(preds, y_batch)

            test_loss += loss.item() * X_batch.size(0)
            test_preds.append(preds.cpu())
            test_targets.append(y_batch.cpu())

            batch_score = nasa_score_tensor(a1, a2, preds, y_batch)
            test_nasa_total += batch_score.item()

    test_loss /= len(test_loader.dataset)
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    test_rmse = torch.sqrt(torch.mean((test_preds - test_targets) ** 2))
    test_nasa = test_nasa_total / len(test_loader.dataset)
    test_time = time.time() - start_time

    print(
        f"Test Loss: {test_loss:.4f} - Test RMSE: {test_rmse:.4f} - "
        f"Test NASA mean: {test_nasa:.4f} - Test NASA total: {test_nasa_total:.2f} - "
        f"Test Time: {test_time:.2f}s"
    )

    return test_loss, test_rmse, test_nasa, test_preds, test_targets, test_time

#========================================================================================================================================


def test_ssl(model, loader, device):

    start_time = time.time()

    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_mae = 0.0
        n_batches = 0

        for x in loader:
            x = x.to(device).float()

            pred, mask, _ = model(x)
            loss = masked_mse(pred, x, mask)
            mae = masked_mae(pred, x, mask)

            running_loss += loss.item()
            running_mae += mae.item()
            n_batches += 1

    elapsed = time.time() - start_time

    return running_loss / max(n_batches, 1), running_mae / max(n_batches, 1), elapsed

#========================================================================================================================================


@torch.no_grad()
def test_rul(model, loader, device):
    
    model.eval()
    model.likelihood.eval()
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_layer, num_data=len(loader.dataset))

    running_loss = 0.0
    n_batches = 0
    all_preds, all_targets, all_vars = [], [], []
    start_time = time.time()

    with gpytorch.settings.fast_pred_var():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float().squeeze(-1)

            output = model(x)
            loss = -mll(output, y)

            running_loss += loss.item()
            n_batches += 1

            posterior = model.likelihood(output)
            all_preds.append(posterior.mean.cpu())
            all_vars.append(posterior.variance.cpu())
            all_targets.append(y.cpu())

    elapsed = time.time() - start_time
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_std = torch.cat(all_vars).clamp_min(0).sqrt()
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

    a1 = 13.0
    a2 = 10.0
    nasa_total = nasa_score_tensor(a1, a2, all_preds, all_targets).item()
    nasa_mean = nasa_total / len(loader.dataset)

    print(
        f"[RUL Test] Loss: {running_loss / max(n_batches, 1):.4f} - RMSE: {rmse:.4f} - "
        f"R2: {r2_score_tensor(all_preds, all_targets):.4f} - "
        f"NASA mean: {nasa_mean:.4f} - NASA total: {nasa_total:.2f} - Time: {elapsed:.2f}s"
    )

    return {
        "loss": running_loss / max(n_batches, 1),
        "rmse": rmse,
        "r2": r2_score_tensor(all_preds, all_targets),
        "nasa_mean": nasa_mean,
        "nasa_total": nasa_total,
        "time": elapsed,
        
        "preds": all_preds.numpy(),
        "targets": all_targets.numpy(),
        "std": all_std.numpy(),
    }


#========================================================================================================================================


@torch.no_grad()
def test_cls(model, loader, device):
    
    model.eval()
    model.likelihood.eval()
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_layer, num_data=len(loader.dataset))

    running_loss = 0.0
    n_batches = 0
    all_probs, all_targets = [], []
    start_time = time.time()

    with gpytorch.settings.fast_pred_var():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float().squeeze(-1)

            output = model(x)
            loss = -mll(output, y)

            running_loss += loss.item()
            n_batches += 1

            probs = model.likelihood(output).mean
            all_probs.append(probs.cpu())
            all_targets.append(y.cpu())

    elapsed = time.time() - start_time
    all_probs = torch.cat(all_probs).numpy()
    all_targets = torch.cat(all_targets).numpy()
    preds = (all_probs >= 0.5).astype(np.float32)

    metrics = {
        "loss": running_loss / max(n_batches, 1),
        "accuracy": accuracy_score(all_targets, preds),
        "f1": f1_score(all_targets, preds, zero_division=0),
        "precision": precision_score(all_targets, preds, zero_division=0),
        "recall": recall_score(all_targets, preds, zero_division=0),
        "brier": brier_score_loss(all_targets, all_probs),
        "ece": expected_calibration_error(all_probs, all_targets),
        "time": elapsed,
    }

    metrics["auroc"] = roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else float("nan")

    print(
        f"[CLS Test] Loss: {metrics['loss']:.4f} - Accuracy: {metrics['accuracy']:.4f} - "
        f"F1: {metrics['f1']:.4f} - Precision: {metrics['precision']:.4f} - Recall: {metrics['recall']:.4f} - "
        f"AUROC: {metrics['auroc']:.4f} - Brier: {metrics['brier']:.4f} - ECE: {metrics['ece']:.4f} - "
        f"Time: {elapsed:.2f}s"
    )

    
    metrics["probs"] = all_probs
    metrics["targets"] = all_targets

    return metrics