import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from module_5.SoftmaxRegression import SoftmaxRegression


def run():
    dataset_path = '../data/creditcard.csv'
    df = pd.read_csv(dataset_path)

    dataset_arr = df.to_numpy()
    X, y = dataset_arr[:, : -1].astype(np.float64), dataset_arr[:, -1].astype(np.uint8)

    intercept = np.ones((X.shape[0], 1))
    x_data = np.concatenate((intercept, X), axis=1)

    n_classes = np.unique(y, axis=0).shape[0]
    n_samples = y.shape[0]
    y_encoded = np.array([np.zeros(n_classes) for _ in range(n_samples)])
    y_encoded[np.arange(n_samples), y] = 1

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True
    x_train, x_val, y_train, _ = train_test_split(
        x_data, y,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle)
    x_train, x_test, y_train, _ = train_test_split(
        x_train, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle)

    normalizer = StandardScaler()
    x_train[:, 1:] = normalizer.fit_transform(x_train[:, 1:])
    x_val[:, 1:] = normalizer.transform(x_val[:, 1:])
    x_test[:, 1:] = normalizer.transform(x_test[:, 1:])

    lr = 0.01
    epochs = 30
    batch_size = 1024
    n_features = x_train.shape[1]

    generator = np.random.default_rng(random_state)
    theta = generator.uniform(size=(n_features, x_train.shape[1]))

    # Training with mini batch
    model_mini_batch = SoftmaxRegression(x_train, y_train, theta=theta,
                                         batch_size=batch_size, learning_rate=lr, num_epochs=epochs)
    model_mini_batch.mini_batch_training()
    model_mini_batch.plot_loss_accuracy()
    print(model_mini_batch.accuracy_model())

