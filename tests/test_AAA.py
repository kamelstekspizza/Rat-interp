import sys
import numpy as np
import scipy.linalg as sl

sys.path.append('../src/')
from rational_interpolation import AAA

def test_sin():
    x = np.linspace(1,3,20)
    y = np.sin(x)

    test_AAA = AAA(x,y)

    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    return


test_sin()