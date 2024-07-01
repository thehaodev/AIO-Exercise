def load_vocab(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
        words = sorted(set([line.strip().lower() for line in lines]))

        return words
