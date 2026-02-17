import numpy as np


ROLL_NUMBER = 102303170

def get_parameters(r):
    ar = 0.5 * (r % 7)
    br = 0.3 * (r % 5 + 1)
    return ar, br

def transform(x):
    ar, br = get_parameters(ROLL_NUMBER)
    z = x + ar * np.sin(br * x)
    return z, ar, br


if __name__ == "__main__":
    x = np.array([10, 20, 30])
    z, ar, br = transform(x)

    print("ar:", ar)
    print("br:", br)
    print("z:", z)
