import sys
import numpy as np
import scipy.linalg as sl

sys.path.append('../src/')
from rational_interpolation import Thiele


def test_sin():
    x = np.linspace(1,3,20)
    y = np.sin(x)

    test = Thiele(x,y)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')

    x = np.linspace(1,3,300)
    y = np.sin(x)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

def test_tan():
    x = np.linspace(0,2*np.pi,20) + 1j*0.2
    y = np.tan(x)

    test = Thiele(x,y)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')

    x = np.linspace(0,2*np.pi,300) + 1j*0.2
    y = np.tan(x)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

def test_rational():
    #Thiele (and also MTT) fails when interpolation interval is [-1,1]. Why?
    x = np.linspace(0,1)
    y = (x**2+1j)/(x**2-2)

    test = Thiele(x,y)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    
    x = np.linspace(-1,1,300)
    y = (x**2+1j)/(x**2-2)
    r = test.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return


test_sin()
test_tan()
test_rational()