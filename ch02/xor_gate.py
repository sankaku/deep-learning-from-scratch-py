import numpy as np
from and_gate import AND
from or_gate import OR
from nand_gate import NAND

def XOR(x1, x2):
    """returns x1 XOR x2"""
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)


def result(x1, x2):
    return '{0} XOR {1} => {2}'.format(x1, x2, XOR(x1, x2))


if __name__ == '__main__':
    print(result(0, 0))
    print(result(0, 1))
    print(result(1, 0))
    print(result(1, 1))
