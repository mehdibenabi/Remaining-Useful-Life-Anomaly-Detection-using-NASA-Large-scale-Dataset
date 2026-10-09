import time
import torch
import gpytorch
from Models import EarlyStopping
from evaluation import nasa_score_tensor, masked_mse, masked_mae, r2_score_tensor
from Testing import test_ssl

def train(criterion, optimizer, train_loader, val_loader, model, num_epochs=20, model_name="Best_model"):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    print(next(model.parameters()).device)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-7
    )

    early_stopper = EarlyStopping(
        patience=5,
        min_delta=1e-5,
        save_path=f"{model_name}.pt"
    )

    a1 = 13.0
    a2 = 10.0

    final_train_preds, final_train_targets = None, None
    final_val_preds, final_val_targets = None, None
    history = []

    training_start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        train_nasa_total = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True).float().squeeze(-1)

            optimizer.zero_grad()
            preds = model(X_batch).squeeze(-1)

            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            train_preds.append(preds.detach().cpu())
            train_targets.append(y_batch.detach().cpu())

            batch_score = nasa_score_tensor(a1, a2, preds.detach(), y_batch.detach())
            train_nasa_total += batch_score.item()

        train_loss /= len(train_loader.dataset)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_rmse = torch.sqrt(torch.mean((train_preds - train_targets) ** 2))
        train_nasa = train_nasa_total / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_nasa_total = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True).float().squeeze(-1)

                preds = model(X_batch).squeeze(-1)
                loss = criterion(preds, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                val_preds.append(preds.cpu())
                val_targets.append(y_batch.cpu())

                batch_score = nasa_score_tensor(a1, a2, preds, y_batch)
                val_nasa_total += batch_score.item()

        val_loss /= len(val_loader.dataset)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = torch.sqrt(torch.mean((val_preds - val_targets) ** 2))
        val_nasa = val_nasa_total / len(val_loader.dataset)

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"LR: {current_lr:.6f} - "
            f"Train Loss: {train_loss:.4f} - Train RMSE: {train_rmse:.4f} - "
            f"Train NASA mean: {train_nasa:.4f} - Train NASA total: {train_nasa_total:.2f} - "
            f"Val Loss: {val_loss:.4f} - Val RMSE: {val_rmse:.4f} - "
            f"Val NASA mean: {val_nasa:.4f} - Val NASA total: {val_nasa_total:.2f} - "
            f"Epoch Time: {epoch_time:.2f}s"
        )

        final_train_preds, final_train_targets = train_preds, train_targets
        final_val_preds, final_val_targets = val_preds, val_targets
        final_loss = val_loss

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_rmse": train_rmse.item(), "val_rmse": val_rmse.item(),
            "train_nasa": train_nasa, "val_nasa": val_nasa,
            "lr": current_lr, "epoch_time": epoch_time,
        })

        scheduler.step(val_loss)
        early_stopper(val_loss, model)

        if early_stopper.early_stop:
            print("Early stopping triggered")
            break


    if early_stopper.has_saved:
        model.load_state_dict(torch.load(early_stopper.save_path))
    else:
        print(f"Warning: training never improved on the initial loss (likely diverged/NaN) - "
              f"no checkpoint was saved to {early_stopper.save_path}, keeping the last epoch's weights.")
    total_train_time = time.time() - training_start

    return model, final_train_preds, final_train_targets, final_val_preds, final_val_targets, final_loss, total_train_time, history


#========================================================================================================================================



def train_ssl(model, train_loader, val_loader, optimizer, epochs=20, device="cuda", model_name="best_ssl_backbone"):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7
    )
    early_stopper = EarlyStopping(patience=5, min_delta=1e-5, save_path=f"{model_name}.pt")
    history = []

    training_start = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        n_batches = 0

        for x in train_loader:
            x = x.to(device).float()
            optimizer.zero_grad()
            pred, mask, _ = model(x)
            loss = masked_mse(pred, x, mask)
            mae = masked_mae(pred, x, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            running_mae += mae.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        train_mae = running_mae / max(n_batches, 1)
        val_loss, val_mae, val_time = test_ssl(model, val_loader, device)
        epoch_time = time.time() - epoch_start

        print(f"[SSL] Epoch {epoch+1} | train_mse={train_loss:.5f} train_mae={train_mae:.5f} | "
              f"val_mse={val_loss:.5f} val_mae={val_mae:.5f} | "
              f"epoch_time={epoch_time:.2f}s val_time={val_time:.2f}s")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_mae": train_mae, "val_mae": val_mae,
            "epoch_time": epoch_time, "val_time": val_time,
        })

        scheduler.step(val_loss)
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print("early stopping triggered")
            break

    if early_stopper.has_saved:
        model.load_state_dict(torch.load(early_stopper.save_path))
    else:
        print(f"Warning: SSL pretraining never improved on the initial loss (likely diverged/NaN) - "
              f"no checkpoint was saved to {early_stopper.save_path}, keeping the last epoch's weights.")
    total_train_time = time.time() - training_start

    return train_loss, train_mae, val_loss, val_mae, total_train_time, history


#========================================================================================================================================


def train_rul(model, loader, optimizer, device):
    
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_layer, num_data=len(loader.dataset))

    a1 = 13.0
    a2 = 10.0

    running_loss = 0.0
    n_batches = 0
    nasa_total = 0.0
    all_preds, all_targets = [], []
    start_time = time.time()

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float().squeeze(-1)

        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            preds = model.likelihood(output).mean
        preds = preds.detach()
        y = y.detach()
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

        nasa_total += nasa_score_tensor(a1, a2, preds, y).item()

    elapsed = time.time() - start_time
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    nasa_mean = nasa_total / len(loader.dataset)

    print(
        f"[RUL Train] Loss: {running_loss / max(n_batches, 1):.4f} - RMSE: {rmse:.4f} - "
        f"NASA mean: {nasa_mean:.4f} - NASA total: {nasa_total:.2f} - Time: {elapsed:.2f}s"
    )

    return {
        "loss": running_loss / max(n_batches, 1),
        "rmse": rmse,
        "r2": r2_score_tensor(all_preds, all_targets),
        "nasa_mean": nasa_mean,
        "nasa_total": nasa_total,
        "time": elapsed,
    }


#========================================================================================================================================


def train_cls(model, loader, optimizer, device):
    
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.VariationalELBO(model.likelihood, model.gp_layer, num_data=len(loader.dataset))

    running_loss = 0.0
    n_batches = 0
    start_time = time.time()

    for x, y in loader:
        x = x.to(device).float()
        y = y.to(device).float().squeeze(-1)

        optimizer.zero_grad()
        output = model(x)
        loss = -mll(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    elapsed = time.time() - start_time
    avg_loss = running_loss / max(n_batches, 1)

    print(f"[CLS Train] Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s")

    return {
        "loss": avg_loss,
        "time": elapsed,
    }
