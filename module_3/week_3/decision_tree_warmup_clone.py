import numpy as np
import pandas as pd
from sklearn import tree
from matplotlib import pyplot as plt


def show_gini_impurity_measures():
    plt.figure()
    x = np.linspace(0.01, 1)
    y = 1 - (x*x) - (1-x)*(1-x)
    plt.plot(x, y)
    plt.title("Gini Impurity")
    plt.xlabel("Fraction of Class K ($p_k$)")
    plt.ylabel("Impurity Measure")
    plt.xticks(np.arange(0, 1.1, 0.1))

    plt.show()


def defining_simple_dataset():
    # Defining a simple dataset
    attribute_names = ['love_math', 'love_art', 'love_english']
    class_name = 'love_ai'
    data = {
        'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
        'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
        'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
        'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no']}
    df = pd.DataFrame(data, columns=data.keys())
    print(df)

    return attribute_names, class_name, df


def gini_impurity(value_counts):
    n = value_counts.sum()
    p_sum = 0
    for key in value_counts.keys():
        p_sum = p_sum + (value_counts[key] / n) * (value_counts[key] / n)

    gini = 1 - p_sum

    return gini


def gini_split_a(attribute_name, df, class_name):
    attribute_values = df[attribute_name].value_counts()
    gini_a = 0

    for key in attribute_values.keys():
        df_k = df[class_name][df[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = df.shape[0]
        gini_a = gini_a + ((n_k / n) * gini_impurity(df_k))

    return gini_a


def run_simple_test():
    # STEP 1: Calculate entire gini(D)
    attribute_names, class_name, df = defining_simple_dataset()
    class_value_counts = df[class_name].value_counts()
    gini_class = gini_impurity(class_value_counts)

    print(f'Number of samples in each class is:\n{class_value_counts}')
    print(f'\nGini Impurity of the class is {gini_class:.3f}')

    # STEP 2:
    # Calculating  gini impurity for the attribute
    gini_attribute = {}
    for key in attribute_names:
        gini_attribute[key] = gini_split_a(key, df, class_name)
        print(f"Gini for {key} is {gini_attribute[key]:.3f}")

    # STEP 3:
    # Compute Gini gain values to find the best split
    # An attribute has maximum Gini gain is selected for splitting.
    min_value = min(gini_attribute.values())
    selected_attribute = min(gini_attribute.keys())
    print('The minimum value of Gini Impurity : {0:.3} '.format(min_value))
    print('The maximum value of Gini Gain     : {0:.3} '.format(1 - min_value))
    print('The selected attribute ís: ', selected_attribute)

    df = df[df[selected_attribute] == 'yes']
    df = df.drop(labels=selected_attribute, axis=1)

    attribute_names.remove(selected_attribute)
    class_name = 'love_ai'

    gini_attribute = {}
    for key in attribute_names:
        gini_attribute[key] = gini_split_a(key, df, class_name)
        print(f'Gini for {key} is {gini_attribute[key]:.3f}')


def run_simple_test_sklearn():
    _, _, df = defining_simple_dataset()
    classifier = tree.DecisionTreeClassifier(criterion="gini",
                                             max_depth=4, min_samples_leaf=1)
    one_hot_data = pd.get_dummies(df[["love_math", "love_art",
                                      "love_english"]], drop_first=True)

    x = one_hot_data.iloc[:, :].values
    y = df['love_ai'].values

    classifier.fit(x, y)

    _, ax = plt.subplots(figsize=(10, 10))
    tree.plot_tree(classifier, ax=ax, feature_names=["love_math", "love_art",
                                                     "love_english"],
                   filled=True)

    plt.show()
    result = classifier.predict([[1, 1, 0]])
    print(result)


run_simple_test_sklearn()




