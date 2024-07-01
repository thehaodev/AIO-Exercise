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
