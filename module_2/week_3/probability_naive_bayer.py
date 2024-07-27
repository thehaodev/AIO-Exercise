import numpy as np


def create_data():
    return np.array([
        ['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'],
        ['Sunny', 'Hot', 'High', 'Weak', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Weak', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
    ], dtype='<U11')


def p_cla_event(feature: int, event: str, sample_space: np.array):
    e_space = sample_space[:, feature]
    events = np.argwhere(e_space == event)

    return events.size / e_space.size


def p_intersect_event(
        feature_a: int,
        feature_b: int,
        event_a,
        event_b,
        sample_space: np.array):
    a_space = sample_space[:, feature_a]
    b_space = sample_space[:, feature_b]
    occur_a_b = 0
    for idx, event in enumerate(a_space):
        if b_space[idx] == event_b and event == event_a:
            occur_a_b += 1

    return occur_a_b / a_space.size


def p_con_event(
        f_e_occur: int,
        f_e: int,
        e,
        e_occur,
        s_space: np.array):
    p_a_and_b = p_intersect_event(feature_a=f_e,
                                  event_a=e,
                                  feature_b=f_e_occur,
                                  event_b=e_occur,
                                  sample_space=s_space)
    p_b = p_cla_event(feature=f_e_occur, event=e_occur, sample_space=s_space)

    return p_a_and_b / p_b


def p_con_multi_event(multi_e: dict, e_occur: str, s_space: np.array, f_e_occur: int, features: dict):
    #
    p_multi_e = 1
    for e in multi_e:
        p_multi_e *= p_con_event(f_e=features[e], e=multi_e[e],
                                 f_e_occur=f_e_occur, e_occur=e_occur,
                                 s_space=s_space)

    return p_multi_e


def p_bayer_multi(e_occur: dict, f_e_predict: int, e_predict: np.array, s_space: np.array, features: dict):
    p_e_occur = 1
    arr_map_e = {}
    arr_p_e_predict = []
    for e in e_predict:
        # Calculate probability for P(e_occur_i|e_predict)
        p_e_predict = p_cla_event(feature=f_e_predict, event=e, sample_space=s_space)
        p_input_e = p_con_multi_event(multi_e=e_occur, f_e_occur=f_e_predict,
                                      e_occur=e, s_space=s_space,
                                      features=features)

        # Add MAP in to array and sum P(e_occur)
        arr_map_e[e] = p_input_e * p_e_predict
        arr_p_e_predict.append(p_e_predict)

        p_e_occur += p_input_e * p_e_predict

    return arr_map_e, p_e_occur


def get_all_class_predict(feature_event_predict: int, sample_space: np.array):
    e_predict_space = sample_space[:, feature_event_predict]
    class_predict = np.unique(np.delete(e_predict_space, 0))

    return class_predict


def exercise_binary_classification():
    # Input data for exercise
    data = create_data()
    event_occur = {
        "Outlook": "Sunny",
        "Temperature": "Cool",
        "Humidity": "High",
        "Wind": "Strong"
    }
    feature_event_predict = "PlayTennis"

    # Get features and event predict
    features = get_feature(data)
    event_predict = get_all_class_predict(features[feature_event_predict], data)
    new_data = np.delete(data, 0, 0)
    arr_map_e, p_e_occur = p_bayer_multi(e_occur=event_occur, f_e_predict=features[feature_event_predict],
                                         e_predict=event_predict, s_space=new_data, features=features)

    for e in event_predict:
        p_e = p_cla_event(feature=features[feature_event_predict], event=e, sample_space=new_data)
        print(f"P({feature_event_predict}={e}) = {p_e}")
    print("")

    # 6, 7, 8, 9
    p_e_occur_predict = {}
    for m in arr_map_e:
        p_e_occur_predict[f"{m}|X"] = arr_map_e[m] / p_e_occur
        print(f"P({feature_event_predict}={m}|X) = {arr_map_e[m]}")
    print("")

    for p in p_e_occur_predict:
        print(f"P({p}) = {p_e_occur_predict[p]}")
    print("")


ON_TIME = 'On Time'
LATE = 'Late'
VERY_LATE = 'Very Late'
CANCELLED = 'Cancelled'


def multi_label_classification():
    # Input data for exercise

    data = np.array([
        ['Day', 'Season', 'Fog', 'Rain', 'Class'],
        ['Weekday', 'Spring', 'None', 'None', ON_TIME],
        ['Weekday', 'Winter', 'None', 'Slight', ON_TIME],
        ['Weekday', 'Winter', 'None', 'None', ON_TIME],
        ['Holiday', 'Winter', 'High', 'Slight', LATE],
        ['Saturday', 'Summer', 'Normal', 'None', ON_TIME],
        ['Weekday', 'Autumn', 'Normal', 'None', VERY_LATE],
        ['Holiday', 'Summer', 'High', 'Slight', ON_TIME],
        ['Sunday', 'Summer', 'Normal', 'None', ON_TIME],
        ['Weekday', 'Winter', 'High', 'Heavy', VERY_LATE],
        ['Weekday', 'Summer', 'None', 'Slight', ON_TIME],
        ['Saturday', 'Spring', 'High', 'Heavy', CANCELLED],
        ['Weekday', 'Summer', 'High', 'Slight', ON_TIME],
        ['Weekday', 'Winter', 'Normal', 'None', LATE],
        ['Weekday', 'Summer', 'High', 'None', ON_TIME],
        ['Weekday', 'Winter', 'Normal', 'Heavy', VERY_LATE],
        ['Saturday', 'Autumn', 'High', 'Slight', ON_TIME],
        ['Weekday', 'Autumn', 'None', 'Heavy', ON_TIME],
        ['Holiday', 'Spring', 'Normal', 'Slight', ON_TIME],
        ['Weekday', 'Spring', 'Normal', 'None', ON_TIME],
        ['Weekday', 'Spring', 'Normal', 'Heavy', ON_TIME]
    ], dtype='<U9')
    event_occur = {
        "Day": "Weekday",
        "Season": "Winter",
        "Fog": "High",
        "Rain": "Heavy"
    }
    feature_event_predict = "Class"

    # Get features and event predict
    features = get_feature(data)
    event_predict = get_all_class_predict(features[feature_event_predict], data)
    new_data = np.delete(data, 0, 0)
    arr_map_e, p_e_occur = p_bayer_multi(e_occur=event_occur, f_e_predict=features[feature_event_predict],
                                         e_predict=event_predict, s_space=new_data, features=features)

    for e in event_predict:
        p_e = p_cla_event(feature=features[feature_event_predict], event=e, sample_space=new_data)
        print(f"P({feature_event_predict}={e}) = {p_e}")
    print("")

    # 6, 7, 8, 9
    p_e_occur_predict = {}
    for m in arr_map_e:
        p_e_occur_predict[f"{m}|X"] = arr_map_e[m] / p_e_occur
        print(f"P({feature_event_predict}={m}|X) = {arr_map_e[m]}")
    print("")

    for p in p_e_occur_predict:
        print(f"P({p}) = {p_e_occur_predict[p]}")


def get_feature(data: np.array):
    feature = {}
    feature_vector = data[0, :]
    for idx, f in enumerate(feature_vector):
        feature[f] = idx

    return feature


def run():
    exercise_binary_classification()
    multi_label_classification()


run()
