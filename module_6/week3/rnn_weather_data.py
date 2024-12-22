import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from module_6 import util


def slicing_window(df, df_start_idx, df_end_idx, input_size, label_size, offset):
    features = []
    labels = []

    window_size = input_size + offset

    if df_end_idx is None:
        df_end_idx = len(df) - window_size

    for idx in range(df_start_idx, df_end_idx):
        feature_end_idx = idx + input_size
        label_start_idx = idx + window_size - label_size

        feature = df[idx:feature_end_idx]
        label = df[label_start_idx:(idx + window_size)]

        features.append(feature)
        labels.append(label)

    features = np.expand_dims(np.array(features), -1)
    labels = np.array(labels)

    return features, labels


class WeatherForecast(Dataset):
    def __init__(
            self,
            x, y,
            transform=None
    ):
        self.X = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.y[idx]

        if self.transform:
            X = self.transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


class WeatherForecastor(nn.Module):
    def __init__(
        self, embedding_dim, hidden_size,
        n_layers, dropout_prob
    ):
        super(WeatherForecastor, self).__init__()
        self.rnn = nn.RNN(
            embedding_dim, hidden_size,
            n_layers, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def run():
    seed = 1
    torch.manual_seed(seed)

    dataset_filepath = 'weatherHistory.csv'
    df = pd.read_csv(dataset_filepath)

    univariate_df = df['Temperature (C)']
    univariate_df.index = df['Formatted Date']
    input_size = 6
    label_size = 1
    offset = 1

    dataset_length = len(univariate_df)
    train_size = 0.7
    val_size = 0.2
    train_end_idx = int(train_size * dataset_length)
    val_end_idx = int(val_size * dataset_length) + train_end_idx

    X_TRAIN, y_train = slicing_window(
        univariate_df,
        df_start_idx=0,
        df_end_idx=train_end_idx,
        input_size=input_size,
        label_size=label_size,
        offset=offset
    )

    X_VAL, y_val = slicing_window(
        univariate_df,
        df_start_idx=train_end_idx,
        df_end_idx=val_end_idx,
        input_size=input_size,
        label_size=label_size,
        offset=offset
    )

    X_TEST, y_test = slicing_window(
        univariate_df,
        df_start_idx=val_end_idx,
        df_end_idx=None,
        input_size=input_size,
        label_size=label_size,
        offset=offset
    )

    train_dataset = WeatherForecast(
        X_TRAIN, y_train
    )
    val_dataset = WeatherForecast(
        X_VAL, y_val
    )
    test_dataset = WeatherForecast(
        X_TEST, y_test
    )

    train_batch_size = 128
    test_batch_size = 8

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )
    _ = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0
    )

    embedding_dim = 1
    hidden_size = 8
    n_layers = 3
    dropout_prob = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = WeatherForecastor(
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        n_layers=n_layers,
        dropout_prob=dropout_prob
    ).to(device)

    lr = 1e-3
    epochs = 50

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0
    )
    _, _ = util.fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs
    )


run()
