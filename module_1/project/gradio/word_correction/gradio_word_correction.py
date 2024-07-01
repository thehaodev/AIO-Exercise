import gradio as gr
from module_1.project.utils.utils_word import load_vocab
from module_1.week_2.levenshtein_distance import find_levenshtein_distance


def display_dict(dictionary):
    # Convert dictionary to formatted string
    output_str = "\n".join([f"{key}: {value}" for key, value in dictionary.items()])
    return output_str


def process(word):
    vocabs = load_vocab(file_path="D:/AI_VIETNAM/CODE_EXERCISE/AIO-Exercise/module_1/project/vocab.txt")
    leven_disntances = dict()
    for vocab in vocabs:
        leven_disntances[vocab] = find_levenshtein_distance(word, vocab)

    sorted_distances = dict(sorted(leven_disntances.items(), key=lambda item: item[1]))
    correct_word = list(sorted_distances.keys())[0]
    dict_string = display_dict(sorted_distances)

    return correct_word, dict_string, dict_string


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Word Correction using Levenshtein Distance
        Word:
        """)
    gr.Interface(fn=process,
                 inputs=[gr.Textbox(label="Input word")],
                 outputs=[gr.Textbox(label="Correct word"),
                          gr.Textbox(label="Vocabulary"),
                          gr.Textbox(label="Distances")])

demo.launch()
