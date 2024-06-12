def find_list_max_numb(list_numb: list, k: int):
    # Check condition
    if not all(isinstance(x, int) for x in list_numb):
        return print("List must contain number only")

    list_result = []
    if k >= len(list_numb):
        list_result.append(max(list_numb))
        return print(list_result)

    for i in range(len(list_numb) - k + 1):
        list_result.append(max_in_range(i, k, list_numb))

    return print(list_result)


def max_in_range(first: int, distance: int, list_sub: list[int]):
    max_numb = list_sub[first]
    for i in range(first, distance+first):
        if list_sub[i] > max_numb:
            max_numb = list_sub[i]

    return max_numb

