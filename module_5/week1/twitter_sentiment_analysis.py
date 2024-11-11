import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from nltk.tokenize import TweetTokenizer
from collections import defaultdict
from module_5.LogisticRegression import LogisticRegression


def text_normalize(text):
    # Retweet old acronym "RT" removal
    text = re.sub(r'RT\s+', '', text)

    # Hyperlinks removal
    text = re.sub(r'https?://.*[\r\n]*', '', text)

    # Hashtags removal
    text = str.replace(r'#', '', text)

    # Punctuation removal
    text = re.sub(r'[^A-Za-z\s]', '', text)

    # Tokenization
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True
    )
    text_tokens = tokenizer.tokenize(text)

    return text_tokens


def get_feature(text, freqs):
    tokens = text_normalize(text)

    X = np.zeros(3)
    X[0] = 1

    for token in tokens:
        X[1] += freqs[(token, 0)]
        X[2] += freqs[(token, 1)]
    return X


def get_freqs(df):
    freqs = defaultdict(lambda: 0)
    for idx, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']

        tokens = text_normalize(tweet)
        for token in tokens:
            pair = (token, label)
            freqs[pair] += 1
    return freqs


def run():
    dataset_path = 'sentiment_analysis.csv'
    df = pd.read_csv(dataset_path, index_col='id')

    X = []
    y = []
    freqs = get_freqs(df)
    for idx, row in df.iterrows():
        tweet = row['tweet']
        label = row['label']
        X_I = get_feature(tweet, freqs)
        X.append(X_I)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    val_size = 0.2
    test_size = 0.125
    random_state = 2
    is_shuffle = True

    X_TRAIN, X_VAL, y_train, _ = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    X_TRAIN, X_TEST, y_train, _ = train_test_split(
        X_TRAIN, y_train,
        test_size=test_size,
        random_state=random_state,
        shuffle=is_shuffle
    )

    normalizer = StandardScaler()
    X_TRAIN[:, :] = normalizer.fit_transform(X_TRAIN[:, :])
    X_VAL[:, :] = normalizer.transform(X_VAL[:, :])
    X_TEST[:, :] = normalizer.transform(X_TEST[:, :])

    lr = 0.01
    epochs = 100
    batch_size = 16

    generator = np.random.default_rng(random_state)
    theta = generator.uniform(size=X_TRAIN.shape[1])

    # Training with mini batch
    model_mini_batch = LogisticRegression(X_TRAIN, y_train, theta=theta,
                                          batch_size=batch_size, learning_rate=lr, num_epochs=epochs)
    model_mini_batch.mini_batch_training()
    model_mini_batch.plot_loss_accuracy()
    print(model_mini_batch.accuracy_model())





