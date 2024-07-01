import streamlit as st


def find_levenshtein_distance(source: str, target: str):
    # Create matrix with two space insert
    first_row = [_ for _ in f"  {source}"]
    first_col = [_ for _ in f"  {target}"]
    rows: int = len(first_col)
    cols: int = len(first_row)
    matrix = [[0] * cols for _ in range(rows)]

    # Fill the first row with source char and first column with target char
    for c in range(cols):
        matrix[0][c] = first_row[c]
    for r in range(rows):
        matrix[r][0] = first_col[r]

    # Numbering the initial transformations
    for i in range(2, cols):
        matrix[1][i] = i-1
    for i in range(2, rows):
        matrix[i][1] = i-1

    # Numbering the transformations to trans from source to target
    for r in range(2, rows):
        for c in range(2, cols):
            sub_cost = 0
            if matrix[0][c] != matrix[r][0]:
                sub_cost = 1
            matrix[r][c] = min(matrix[r][c-1] + 1, matrix[r-1][c] + 1, matrix[r-1][c-1] + sub_cost)

    return matrix[rows-1][cols-1]


def load_vocab(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = sorted(set([line.strip().lower() for line in lines]))

        return words


def run():
    st.title("Word Correction using Levenshtein Distance")
    word = st.text_input("Word: ")
    vocabs = load_vocab(file_path="module_1/project/vocab.txt")

    if st.button("Compute"):
        leven_disntances = dict()
        for vocab in vocabs:
            leven_disntances[vocab] = find_levenshtein_distance(word, vocab)

        sorted_distances = dict(sorted(leven_disntances.items(), key=lambda item: item[1]))
        correct_word = list(sorted_distances.keys())[0]
        st.write("Correct word: ", correct_word)

        col1, col2 = st.columns(2)
        col1.write("Vocabulary: ")
        col1.write(vocabs)

        col2.write("Distances: ")
        col2.write(sorted_distances)


run()
