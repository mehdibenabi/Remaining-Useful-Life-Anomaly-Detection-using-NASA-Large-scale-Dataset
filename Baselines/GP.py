import time
import torch
import gpytorch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# ================================================================================
# CONFIGURATION
# ================================================================================
LEARNING_RATE = 0.01
MAX_NUM_EPOCHS = 300
NUM_INDUCING_POINTS = 400
BATCH_SIZE = 1024
RANDOM_STATE = 0
VERBOSE = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ================================================================================
# EARLY STOPPING
# ================================================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def check_early_stopping(self, epoch, loss):
        if self.best_loss is None:
            self.best_loss = loss
        elif loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                self.should_stop = True
                return True
        return False


# ================================================================================
# SVGP MODEL
# ================================================================================
class SVGPRegressionModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ================================================================================
# TRAINER
# ================================================================================
class SVGPTrainer:
    def __init__(self, num_inducing=NUM_INDUCING_POINTS, learning_rate=LEARNING_RATE,
                 max_epochs=MAX_NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE):
        self.num_inducing = num_inducing
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = device

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if self.verbose:
            print('=' * 60)
            print("TRAINING SVGP MODEL")
            print('=' * 60)
            print(f"Training samples: {len(X_train)}")
            print(f"Inducing points: {self.num_inducing}")

        start_time = time.time()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)

        # Initialize inducing points with k-means 
        subsample_n = min(20000, len(X_train))
        idx = np.random.choice(len(X_train), subsample_n, replace=False)
        kmeans = KMeans(n_clusters=self.num_inducing, init='k-means++',
                         random_state=RANDOM_STATE, n_init=10).fit(X_train[idx])
        inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        model = SVGPRegressionModel(inducing_points).to(self.device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model.parameters()},
            {'params': likelihood.parameters()}
        ], lr=self.learning_rate)

        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_tensor.size(0))

        # Mini-batch training via TensorDataset and DataLoader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True)

        es = EarlyStopping(patience=15, min_delta=1e-4)

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)
            epoch_loss /= len(train_dataset)

            # Optional validation RMSE tracking
            if X_val is not None and (epoch % 5 == 0 or epoch == self.max_epochs - 1):
                val_rmse = self._quick_eval(model, likelihood, X_val, y_val)
                if self.verbose:
                    print(f"Epoch {epoch:4d} | Train ELBO loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.4f}")
            elif self.verbose and epoch % 20 == 0:
                print(f"Epoch {epoch:4d} | Train ELBO loss: {epoch_loss:.4f}")

            if es.check_early_stopping(epoch, epoch_loss):
                break

        training_time = time.time() - start_time
        if self.verbose:
            print(f"✓ Training completed in {training_time:.2f}s")

        return model, likelihood, training_time

    def _quick_eval(self, model, likelihood, X_val, y_val):
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            preds = likelihood(model(X_val_tensor)).mean.cpu().numpy()
        model.train()
        likelihood.train()
        return np.sqrt(mean_squared_error(y_val, preds))

    def predict(self, model, likelihood, X_test, batch_size=4096):
        """Predict mean and variance in batches (avoids huge kernel matrix at once)"""
        model.eval()
        likelihood.eval()
        means, variances = [], []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for start in range(0, len(X_test), batch_size):
                end = start + batch_size
                X_batch = torch.tensor(X_test[start:end], dtype=torch.float32).to(self.device)
                pred = likelihood(model(X_batch))
                means.append(pred.mean.cpu().numpy())
                variances.append(pred.variance.cpu().numpy())

        return np.concatenate(means), np.concatenate(variances)



# ================================================================================
# NASA SCORING FUNCTION
# ================================================================================
def nasa_score(y_true, y_pred):
    d = y_pred - y_true  # positive = over-estimation, penalized more
    score = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return np.sum(score)


# ================================================================================
# MAIN PIPELINE
# ================================================================================
def run_svgp_pipeline(num_inducing, learning_rate, batch_size,  X_train_gp, Y_windows_train,
                       X_val_gp, Y_windows_val,
                       X_test_gp, Y_windows_test):

    print(f'num_inducing: {num_inducing}')
    trainer = SVGPTrainer(
        num_inducing=num_inducing,
        learning_rate=learning_rate,
        max_epochs=MAX_NUM_EPOCHS,
        batch_size=batch_size,
        verbose=VERBOSE
    )

    model, likelihood, train_time = trainer.train(
        X_train_gp, Y_windows_train, X_val_gp, Y_windows_val
    )

    # Validation evaluation
    val_mean, val_var = trainer.predict(model, likelihood, X_val_gp)
    val_rmse = np.sqrt(mean_squared_error(Y_windows_val, val_mean))
    val_nasa = nasa_score(Y_windows_val, val_mean)

    print('\n' + '=' * 60)
    print("VALIDATION RESULTS")
    print('=' * 60)
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  NASA Score: {val_nasa:.2f}")

    # Test evaluation
    start_infer = time.time()
    test_mean, test_var = trainer.predict(model, likelihood, X_test_gp)
    infer_time = time.time() - start_infer

    test_rmse = np.sqrt(mean_squared_error(Y_windows_test, test_mean))
    test_nasa = nasa_score(Y_windows_test, test_mean)
    test_std = np.sqrt(test_var)

    print('\n' + '=' * 60)
    print("TEST RESULTS")
    print('=' * 60)
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  NASA Score: {test_nasa:.2f}")
    print(f"  Mean predicted std (uncertainty): {test_std.mean():.4f}")
    print(f"  Inference time: {infer_time:.2f}s ({(infer_time/len(X_test_gp))*1000:.3f}ms/sample)")

    return {
        'model': model,
        'likelihood': likelihood,
        'val_rmse': val_rmse,
        'val_nasa': val_nasa,
        'test_rmse': test_rmse,
        'test_nasa': test_nasa,
        'test_mean': test_mean,
        'test_std': test_std,
        'train_time': train_time,
        'infer_time': infer_time
    }



def plot_predictions_gp(y_true, y_pred):
    plt.figure(figsize=(8,6))
    plt.scatter(y_true, y_pred, alpha=0.6)

    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]

    plt.plot(lims, lims, 'r--', linewidth=2)

    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Predicted vs True RUL")
    plt.grid(True)

    plt.show()

def plot_errors(y_true, y_pred):

    errors = y_pred - y_true

    plt.figure(figsize=(10,5))

    plt.plot(errors)

    plt.axhline(0,color='red',linestyle='--')

    plt.xlabel("Sample")
    plt.ylabel("Prediction Error")

    plt.title("Prediction Error")

    plt.grid(True)

    plt.show()

def plot_uncertainty(y_true, y_pred, std):

    plt.figure(figsize=(12,5))

    x = np.arange(len(y_true))

    plt.plot(x,y_true,label="True")

    plt.plot(x,y_pred,label="Prediction")

    plt.fill_between(
        x,
        y_pred-2*std,
        y_pred+2*std,
        alpha=0.3,
        label="95% Confidence"
    )

    plt.legend()

    plt.xlabel("Sample")

    plt.ylabel("RUL")

    plt.title("Prediction with Uncertainty")

    plt.show()