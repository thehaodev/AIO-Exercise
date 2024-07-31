import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from gdown.exceptions import FileURLRetrievalError


def compute_mean(x: np.array):
    return np.mean(x)


def compute_median(x: np.array):
    size = len(x)
    x = np.sort(x)
    if size % 2 == 0:
        return x[int((size + 1) / 2) - 1]
    else:
        return 1 / 2 * (x[int(size / 2 - 1)] + x[int(size / 2)])


def compute_std(x: np.array):
    var = np.var(x)
    return np.sqrt(var)


def compute_correlation_coefficient(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)

    r_numerator = 0
    for idx, _ in enumerate(x):
        r_numerator += (x[idx] - x_m) * (y[idx] - y_m)

    x_var = np.var(x) * len(x)
    y_var = np.var(y) * len(y)
    r_denominator = np.sqrt(x_var * y_var)
    return r_numerator / r_denominator


def tfidf_search(context_embedded, question, tfidf_vectorizer, top_d=5):
    query = question.lower()
    query_embedded = tfidf_vectorizer.transform([query])
    cosine_scores = cosine_similarity(context_embedded, query_embedded)
    results = []
    for idx in cosine_scores.argsort()[- top_d:][:: -1]:
        doc_score = {
            'id ': idx,
            'cosine_score': cosine_scores[idx]
        }
        results.append(doc_score)

    return results


def corr_search(context_embedded, question, tfidf_vectorizer, top_d=5):
    query = question.lower()
    query_embedded = tfidf_vectorizer.transform([query])
    corr_scores = np.corrcoef(query_embedded.toarray(), context_embedded.toarray())
    corr_scores = corr_scores[0][1:]

    results = []
    for idx in corr_scores.argsort()[- top_d:][:: -1]:
        doc = {
            'id ': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)

    return results


def text_retrieval_exercise():
    url = "https://drive.google.com/uc?id=1jh2p2DlaWsDo_vEWIcTrNh3mUuXd-cw6"
    try:
        gdown.download(url)
    except FileURLRetrievalError:
        print("File cannot download")

    vi_data_df = pd.read_csv("./vi_text_retrieval.csv")
    context = vi_data_df['text']
    context = [doc.lower() for doc in context]
    tfidf_vectorizer = TfidfVectorizer()

    # Question 10
    context_embedded = tfidf_vectorizer.fit_transform(context)
    print(context_embedded.toarray()[7][0])

    # Question 11
    question = vi_data_df.iloc[0]['question']
    results = tfidf_search(context_embedded, question, tfidf_vectorizer=tfidf_vectorizer)
    print(results[0]['cosine_score'])

    # Question 12
    results = corr_search(context_embedded, question, tfidf_vectorizer)
    print(results[1]['corr_score'])


def analysis_advertising():
    data = pd.read_csv("advertising.csv")
    list_feature = [col for col in data.columns if col != "species"]
    r_matrix = np.ones((len(list_feature), len(list_feature)))
    for i, f in enumerate(list_feature):
        for j, f_reversed in enumerate(list_feature):
            data_i = data[f]
            data_j = data[f_reversed]
            r_matrix[i][j] = compute_correlation_coefficient(data_i, data_j)

    result = pd.DataFrame(r_matrix, index=list_feature, columns=list_feature)
    plt.figure(figsize=(10, 8))
    sns.heatmap(result, annot=True, fmt=".2f", linewidth=.5)
    plt.show()


def multi_choice_exercise():
    # Question 1
    x = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
    print(compute_mean(x))

    # Question 2
    x = [1, 5, 4, 4, 9, 13]
    print(compute_median(x))

    # Question 3
    x = [171, 176, 155, 167, 169, 182]
    print(compute_std(x))

    # Question 4
    x = np.asarray([-2, -5, -11, 6, 4, 15, 9])
    y = np.asarray([4, 25, 121, 36, 16, 225, 81])
    print(compute_correlation_coefficient(x, y))

    # Question 5 -> 9
    # Note the data handle code can replace with one line data.corr()
    analysis_advertising()


multi_choice_exercise()
text_retrieval_exercise()
