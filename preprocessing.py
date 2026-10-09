import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn import metrics, calibration
from torch.utils.data import Dataset, DataLoader


class CMAPSS(Dataset):
    def __init__(self, x, y):
        self.X = np.asarray(x).astype(np.float32)
        self.y = np.asarray(y).astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class CMAPSS_SSL(Dataset):
    def __init__(self, x):
        self.X = np.asarray(x).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx])


class CMAPSSWindowed(Dataset):
    

    def __init__(self, df, features, window_size, stride, has_target=True, target_col="RUL"):
        self.window_size = window_size
        self.has_target = has_target
        self.sequences = []   # one (T, F) float32 array per unit
        self.targets = []     # one (T,) float32 array per unit 
        self.index = []       # (unit_idx, start) for every window

        for unit_idx, (_, unit_df) in enumerate(df.groupby('unit', sort=False)):
            feature_data = np.ascontiguousarray(unit_df[features].values, dtype=np.float32)
            self.sequences.append(feature_data)

            if has_target:
                self.targets.append(np.ascontiguousarray(unit_df[target_col].values, dtype=np.float32))
            else:
                self.targets.append(None)

            n_timesteps = len(unit_df)
            for start in range(0, n_timesteps - window_size + 1, stride):
                self.index.append((unit_idx, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        unit_idx, start = self.index[idx]
        end = start + self.window_size
        x = torch.from_numpy(self.sequences[unit_idx][start:end].copy())

        if not self.has_target:
            return x

        y = self.targets[unit_idx][end - 1]
        return x, torch.tensor([y], dtype=torch.float32)



def preprocessing(Normalization, df_train, df_test, features):

    

    X_train = df_train[features]
    X_test  = df_test[features]

    Y_train = df_train['RUL']
    Y_test  = df_test['RUL']

    
    has_hs = "hs" in df_train.columns

    if Normalization:

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_normalized_df = scaler.fit_transform(X_train)
        X_test_normalized_df  = scaler.transform(X_test)

        # rebuild as DataFrames with unit/cycle/RUL attached
        X_train_normalized_df = pd.DataFrame(X_train_normalized_df, columns=features, index=df_train.index)
        X_train_normalized_df['unit'] = df_train['unit']
        X_train_normalized_df['cycle'] = df_train['cycle']
        X_train_normalized_df['RUL'] = Y_train
        if has_hs:
            X_train_normalized_df['hs'] = df_train['hs']
        

        X_test_normalized_df = pd.DataFrame(X_test_normalized_df, columns=features, index=df_test.index)
        X_test_normalized_df['unit'] = df_test['unit']
        X_test_normalized_df['cycle'] = df_test['cycle']
        X_test_normalized_df['RUL'] = Y_test
        if has_hs:
            X_test_normalized_df['hs'] = df_test['hs']
        

        print("Data Normalization\n")
        return X_train_normalized_df, X_test_normalized_df, Y_train, Y_test

    else:

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        # rebuild as DataFrames with unit/cycle/RUL attached
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features, index=df_train.index)
        X_train_scaled_df['unit'] = df_train['unit']
        X_train_scaled_df['cycle'] = df_train['cycle']
        X_train_scaled_df['RUL'] = Y_train
        if has_hs:
            X_train_scaled_df['hs'] = df_train['hs']
        

        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=features, index=df_test.index)
        X_test_scaled_df['unit'] = df_test['unit']
        X_test_scaled_df['cycle'] = df_test['cycle']
        X_test_scaled_df['RUL'] = Y_test
        if has_hs:
            X_test_scaled_df['hs'] = df_test['hs']
        

        print("Data standardized\n")
        return X_train_scaled_df, X_test_scaled_df, Y_train, Y_test


def create_windows(df, features, window_size, stride, target_col="RUL"):
   
    X_list = []
    Y_list = []

    for unit_id, unit_df in df.groupby('unit', sort=False):
        feature_data = np.ascontiguousarray(unit_df[features].values, dtype=np.float32)
        target_data = unit_df[target_col].values

        n_timesteps = len(unit_df)
        if n_timesteps < window_size:
            continue

        
        windows = np.lib.stride_tricks.sliding_window_view(feature_data, window_size, axis=0)
        windows = windows.transpose(0, 2, 1)     
        windows = windows[::stride]               

        end_indices = np.arange(window_size - 1, n_timesteps, stride)
        labels = target_data[end_indices]

        X_list.append(windows)
        Y_list.append(labels)

    if not X_list:
        return (np.empty((0, window_size, len(features)), dtype=np.float32),
                np.empty((0,), dtype=np.float32))

    X_windows = np.concatenate(X_list, axis=0)
    Y_windows = np.concatenate(Y_list, axis=0)

    return X_windows, Y_windows
