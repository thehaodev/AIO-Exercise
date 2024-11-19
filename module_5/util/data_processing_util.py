from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sklearn_split_data(val_size, test_size, random_state, x_features, y_label):
    is_shuffle = True
    x_train, x_val, y_train, y_val = train_test_split(
        x_features, y_label,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle)
    x_train, x_test, y_train, y_test = train_test_split(
        x_train, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle)

    return x_train, y_train, x_test, y_test, x_val, y_val


def sklearn_normalizer(x_train, x_val, x_test):
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_val = normalizer.transform(x_val)
    x_test = normalizer.transform(x_test)

    return x_train, x_val, x_test
