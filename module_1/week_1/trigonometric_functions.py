def approx_sin(x: float, n: int):
    sin = 0
    for i in range(n):
        sin += pow(-1, i) * pow(x, 2*i + 1) / factorial(2*i + 1)

    return print(sin)


def approx_cos(x: float, n: int):
    cos = 0
    for i in range(n):
        cos += pow(-1, i) * pow(x, 2*i) / factorial(2*i)

    return print(cos)


def approx_sinh(x: float, n: int):
    sinh = 0
    for i in range(n):
        sinh += pow(x, 2*i + 1) / factorial(2*i + 1)

    return print(sinh)


def approx_cosh(x: float, n: int):
    cosh = 0
    for i in range(n):
        cosh += pow(x, 2*i) / factorial(2*i)

    return print(cosh)


def factorial(n: int):
    fact = 1
    for i in range(1, n+1):
        fact *= i

    return fact
