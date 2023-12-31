import sys
import numpy as np

sys.path.append('../src/')
from rational_interpolation import AAA

def test_sin():
    x = np.linspace(1,3,20)
    y = np.sin(x)

    test_AAA = AAA(x,y)

    return


test_sin()