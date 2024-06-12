import gdown
from gdown.exceptions import FileURLRetrievalError


def read_word_from_file():
    url = "https://drive.google.com/uc?id=1IBScGdW2xlNsc9v5zSAya548kNgiOrko"
    output = "text.txt"
    try:
        gdown.download(url, output)
    except FileURLRetrievalError:
        return print("File cannot download")

    file = open(output, "r")

    list_line = file.readlines()
    dict_word = {}
    if len(list_line) > 0:
        for line in list_line:
            list_word = line.split(" ")
            for word in list_word:
                if word not in dict_word:
                    dict_word.update({word: 1})
                else:
                    dict_word[word] += 1
    else:
        return print("File empty")

    file.close()

    return print(dict_word)
