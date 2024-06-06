def calc_f1_score(true_positives, false_positives, false_negatives):
    if (type(true_positives) is not int
            or type(false_positives) is not int
            or type(false_negatives) is not int):
        if type(true_positives) is not int:
            print("true positives must be int")
        if type(false_positives) is not int:
            print("false positives must be int")
        if type(false_negatives) is not int:
            print("false negatives must be int")
        return

    if true_positives <= 0 or false_positives <= 0 or false_positives <= 0:
        return print("true positives and false positives and false negatives must be greater than zero")

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)

    return print(f"precision is {precision}\n"
                 f"recall is {recall}\n"
                 f"f1_score is {f1_score}")
