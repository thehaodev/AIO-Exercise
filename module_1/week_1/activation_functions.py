import math


def calc_activation_functions():
    x = input("Input x = ")
    if not is_number(x):
        return print("x must be a number")

    functions = ["sigmoid", "relu", "elu"]
    input_funct = input("Input activation Function ( sigmoid | relu | elu ): ")
    if input_funct.lower() not in functions:
        return print(f"{input_funct} is not supported")

    if input_funct.lower() == functions[0]:
        sigmoid = 1 / (1 + pow(math.e, float(x) * (-1)))
        return print(f"sigmoid: f({input_funct}): {sigmoid}")

    if input_funct.lower() == functions[1]:
        if float(x) <= 0:
            return print(f"relu: f({x}) = 0")
        else:
            return print(f"relu: f({x}) = {x}")

    if input_funct.lower() == functions[2]:
        alpha = 0.01

        if float(x) <= 0:
            elu = alpha * (pow(math.e, float(x)) - 1)
            return print(f"elu: f({x}) = {elu})")
        else:
            return print(f"elu: f({x}) = {x}")


def is_number(n):
    try:
        float(n)
    except ValueError:
        return False
    return True
