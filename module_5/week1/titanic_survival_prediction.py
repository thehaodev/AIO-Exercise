import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from module_5.LogisticRegression import LogisticRegression


def run():
    dataset_path = '../data/titanic_modified_dataset.csv'
    df = pd.read_csv(dataset_path, index_col='PassengerId')
    print(df)

    dataset_arr = df.to_numpy().astype(np.float64)
    X, y = dataset_arr[:, :-1], dataset_arr[:, -1]

    intercept = np.ones((X.shape[0], 1))
    x_data = np.concatenate((intercept, X), axis=1)

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
    epochs = 100
    batch_size = 16

    generator = np.random.default_rng(random_state)
    theta = generator.uniform(size=x_train.shape[1])

    # Training with stochastic
    # model_stochastic = LogisticRegression(x_train, y_train, theta=theta,
    #                                       batch_size=batch_size, learning_rate=lr, num_epochs=epochs)
    # model_stochastic.stochastic_training()
    # model_stochastic.plot_loss_accuracy()
    # print(model_stochastic.accuracy_model())

    # Training with mini batch
    model_mini_batch = LogisticRegression(x_train, y_train, theta=theta,
                                          batch_size=batch_size, learning_rate=lr, num_epochs=epochs)
    model_mini_batch.mini_batch_training()
    model_mini_batch.plot_loss_accuracy()
    print(model_mini_batch.accuracy_model())


run()
