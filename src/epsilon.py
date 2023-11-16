import numpy as np

def wynn_epsilon(S_n):
    n = S_n.shape[0]
    epsilon = np.zeros((n+1,n+1),dtype=S_n.dtype)

    epsilon[1,:n] = S_n.copy()

    #Use loops for both iterations
    #for k in range(2,n+1):
    #    for m in range(n+2-k):
    #        epsilon[k,m] = epsilon[k-2,m+1] + 1/(epsilon[k-1,m+1]-epsilon[k-1,m])

    #vectorize inner loop
    for k in range(2,n+1):
        epsilon[k,:n+2-k] = epsilon[k-2,1:n+2-(k-1)] + 1/(epsilon[k-1,1:n+2-(k-1)]-epsilon[k-1,:n+2-k])

    return epsilon

def test_epsilon(n):

    S_n = np.zeros(n)

    S_n[0] = 4.0
    for i in range(1,n):
        S_n[i] = S_n[i-1] + 4.0*(-1.0)**i/(2.0*i+1.0) 


    epsilon = wynn_epsilon(S_n)

    print(epsilon[1::2,:])
    print(np.pi)

