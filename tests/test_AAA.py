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
    print(f'Convergence history: {test_AAA.errvec}')

    x = np.linspace(1,3,300)
    y = np.sin(x)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

def test_tan():
    x = np.linspace(0,2*np.pi,20) + 1j*0.2
    y = np.tan(x)

    test_AAA = AAA(x,y)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    print(f'Convergence history: {test_AAA.errvec}')

    x = np.linspace(0,2*np.pi,300) + 1j*0.2
    y = np.tan(x)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

def test_rational():
    x = np.linspace(-1,1)
    y = (x**2+1j)/(x**2-2)

    test_AAA = AAA(x,y)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    print(f'Convergence history: {test_AAA.errvec}')

    x = np.linspace(-1,1,300)
    y = (x**2+1j)/(x**2-2)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return



def test_sqrt():
    x = np.linspace(0,1,int(2e5))
    y = np.sqrt(x)

    test_AAA = AAA(x,y)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    print(f'Convergence history: {test_AAA.errvec}')

    x = np.linspace(0,1,300)
    y = np.sqrt(x)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

def test_abs():
    x = np.linspace(-1,1,int(2e5))
    y = np.abs(x)

    test_AAA = AAA(x,y)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative interpolation error: {rel_err}')
    print(f'Convergence history: {test_AAA.errvec}')

    x = np.linspace(-1,1,300)
    y = np.abs(x)
    r = test_AAA.eval(x)
    err = sl.norm(r-y,ord=np.inf)
    rel_err = err/sl.norm(y,ord=np.inf)
    print(f'Relative error: {rel_err}')

    return

test_sin()
test_tan()
test_rational()
test_sqrt()
test_abs()