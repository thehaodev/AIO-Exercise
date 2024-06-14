def count_chars(s: str):
    dict_chars = {}
    for i in s:
        if i not in dict_chars:
            dict_chars.update({i: 1})
        else:
            dict_chars[i] += 1

    return print(dict_chars)
