import streamlit as st
from pathlib import Path
from module_1.project.utils.utils_word import load_vocab
from module_1.week_2.levenshtein_distance import find_levenshtein_distance

path = Path(__file__).parent.resolve()


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
