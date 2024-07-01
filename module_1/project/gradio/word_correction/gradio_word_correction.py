import gradio as gr


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
        matrix[1][i] = i - 1
    for i in range(2, rows):
        matrix[i][1] = i - 1

    # Numbering the transformations to trans from source to target
    for r in range(2, rows):
        for c in range(2, cols):
            sub_cost = 0
            if matrix[0][c] != matrix[r][0]:
                sub_cost = 1
            matrix[r][c] = min(matrix[r][c - 1] + 1, matrix[r - 1][c] + 1, matrix[r - 1][c - 1] + sub_cost)

    return matrix[rows - 1][cols - 1]


def load_vocab(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = sorted(set([line.strip().lower() for line in lines]))

        return words


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
