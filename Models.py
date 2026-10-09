import torch
import torch.nn as nn
import gpytorch
from sklearn.cluster import KMeans



class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, save_path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False
        self.has_saved = False

    def __call__(self, val_loss, model):
        if val_loss != val_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            self.has_saved = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


#===============================================================================================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


#===============================================================================================================================================

# Vanilla transformer architecture

class FirstBaseline(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()

        
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        x = self.input_proj(x)          # (batch, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x) # (batch, seq_len, d_model)
        x = x.mean(dim=1)               # mean pooling over time , (batch, d_model)
        out = self.output_head(x)       # (batch, 1)
        return out


#========================================================================================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNRegressor(nn.Module):
    def __init__(self, num_features, num_channels=(64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = num_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        out = self.head(y)
        return out




#========================================================================================================================================
class TransformerBackbone(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        z_seq = self.transformer_encoder(x)   # (B, T, D)
        z_pool = z_seq.mean(dim=1)            # (B, D)
        return z_seq, z_pool


#========================================================================================================================================
class TCNBackbone(nn.Module):
    def __init__(self, num_features, num_channels=(64, 64, 64), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.d_model = num_channels[-1]

    def forward(self, x):
        x = x.transpose(1, 2)          # (B, F, T)
        y = self.tcn(x)                # (B, C, T)
        z_seq = y.transpose(1, 2)      # (B, T, C)
        z_pool = y[:, :, -1]           # (B, C)
        return z_seq, z_pool

#========================================================================================================================================

class TCNTransformerBackbone(nn.Module):
    def __init__(self, num_features, num_channels=(64, 64), kernel_size=3,
                 dropout=0.1, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            in_ch = num_features if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        self.proj = nn.Linear(num_channels[-1], d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x).transpose(1, 2)   # (B, T, C)
        y = self.proj(y)
        y = self.pos_encoder(y)
        z_seq = self.transformer_encoder(y)
        z_pool = z_seq.mean(dim=1)
        return z_seq, z_pool

#========================================================================================================================================

class MaskedReconstructionSSL(nn.Module):
    def __init__(self, backbone, d_model, num_features, mask_ratio=0.3):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio

        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_features)
        )

    def make_mask(self, x):
        return torch.rand_like(x) < self.mask_ratio

    def forward(self, x):
        mask = self.make_mask(x)
        x_masked = x.clone()
        x_masked[mask] = 0.0

        z_seq, z_pool = self.backbone(x_masked)
        pred = self.recon_head(z_seq)
        return pred, mask, z_pool

#========================================================================================================================================

class SVGPLayer(gpytorch.models.ApproximateGP):
  

    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution,
            learn_inducing_locations=True)
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.size(-1)))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def init_inducing_points(backbone, loader, device, num_inducing=100, max_samples=20000, random_state=0):
    """
    Seeds inducing points via k-means over backbone-pooled features.
    """
    backbone.eval()
    pooled = []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device).float()
            _, z_pool = backbone(x)
            pooled.append(z_pool.cpu())
            n_collected += z_pool.size(0)
            if n_collected >= max_samples:
                break

    pooled = torch.cat(pooled, dim=0)[:max_samples].numpy()
    n_clusters = min(num_inducing, len(pooled))
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=10).fit(pooled)
    return torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)


class RULHead(nn.Module):
    

    def __init__(self, backbone, d_model, inducing_points=None, num_inducing=100):
        super().__init__()
        self.backbone = backbone
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing, d_model)
        self.gp_layer = SVGPLayer(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x):
        _, z_pool = self.backbone(x)
        return self.gp_layer(z_pool)

#========================================================================================================================================

class BinaryClassificationHead(nn.Module):
    

    def __init__(self, backbone, d_model, inducing_points=None, num_inducing=100):
        super().__init__()
        self.backbone = backbone
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing, d_model)
        self.gp_layer = SVGPLayer(inducing_points)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, x):
        _, z_pool = self.backbone(x)
        return self.gp_layer(z_pool)

#========================================================================================================================================

def load_backbone_weights(head_model, checkpoint_path, map_location=None, freeze=False):
    
    state_dict = torch.load(checkpoint_path, map_location=map_location)
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in state_dict.items()
        if key.startswith("backbone.")
    }
    head_model.backbone.load_state_dict(backbone_state)

    if freeze:
        for param in head_model.backbone.parameters():
            param.requires_grad = False

    return head_model


  

