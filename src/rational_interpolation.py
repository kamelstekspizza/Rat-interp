#!/usr/bin/python3
#Provides functionality to perform rational interpolation
#Can be used for Cormier-Lambropoulos extrapolation method
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.linalg as sl
from scipy.interpolate import pade

class multi_point_Pade:
    def __init__(self,x,samples):
        if len(x)!= len(samples):
            raise ValueError('The number of interpolation points is different from the number of function samples')
        
        self.x = x.copy()
        self.N = len(samples)
        self.a = np.zeros(self.N,dtype = np.complex128)
        self.a[0] = samples[0]
        g_i_minus = np.zeros(self.N,dtype = np.complex128)
        g_i = np.zeros(self.N,dtype = np.complex128)
        g_i_minus = samples.copy()
        for i in range(1,self.N):
            g_i = (g_i_minus[i-1]-g_i_minus)/((self.x-self.x[i-1])*g_i_minus)
            self.a[i] = g_i[i]
            g_i_minus = g_i.copy()

        #print(self.a)
            
    def val(self,x):
        value = 1.0
        for i in reversed(range(1,self.N)):
            value = 1+ self.a[i]*(x-self.x[i-1])/value
        value = self.a[0]/value
        
        return value
    
    
def thieleInterpolator(x, y):
    ρ = [[yi]*(len(y)-i) for i, yi in enumerate(y)]
    for i in range(len(ρ)-1):
        ρ[i][1] = (x[i] - x[i+1]) / (ρ[i][0] - ρ[i+1][0])
    for i in range(2, len(ρ)):
        for j in range(len(ρ)-i):
            ρ[j][i] = (x[j]-x[j+i]) / (ρ[j][i-1]-ρ[j+1][i-1]) + ρ[j+1][i-2]
    ρ0 = ρ[0]
    def t(xin):
        a = 0
        for i in range(len(ρ0)-1, 1, -1):
            a = (xin - x[i-1]) / (ρ0[i] - ρ0[i-2] + a)
        return y[0] + (xin-x[0]) / (ρ0[1]+a)
    return t

class MTT:
    #Implementation of the Modified Thacher-Tukey algorithm
    #The implementation is based on the description in 
    #Padé approximants (Encyclopedia of Mathematics and its Applications)
    #by George A. Baker Jr. and Peter Graves-Morris.

    def __init__(self,x,samples):
        
        if len(x)!= len(samples):
            raise ValueError('The number of interpolation points is different from the number of function samples')
        
        self.N = len(samples)
        self.x = x.copy()
        self.x_prime = self.x.copy()
        self.f = samples.copy()
        self.t = 0

        self.x_0 = self.x[0]
        self.f_0 = self.f[0]
        self.a = np.zeros(self.N,dtype = self.f.dtype)
        g_j_minus = self.f-self.f_0
        
        for j in range(1,self.N):
            for i in range(j,self.N):
                #Need to implement the conditions for stage (b) and (c)!
                if np.isfinite(g_j_minus[i]) and not np.isclose(g_j_minus[i],0):
                    self.x_prime[[i,j]] = self.x_prime[[j,i]]
                    self.f[[i,j]] = self.f[[j,i]]
                    g_j_minus[[i,j]] =g_j_minus[[j,i]]
                    break
            self.a[j] = g_j_minus[j]/(self.x_prime[j]-self.x_prime[j-1])
            g_j_minus[j+1:] = self.a[j]*(self.x_prime[j+1:]-self.x_prime[j-1])/g_j_minus[j+1:] -1
            
        self.t = self.N-1

        #Termination stage
        if self.t==0:
            #r(z) = f_0
            self.val = self.eval_const
            return
        
        q_1 = np.ones(self.N,dtype=self.f.dtype)
        q_2 = q_1 + self.a[2]*(self.x-self.x_prime[1])

        for i in range(2,self.t):
            q_temp = q_2 + self.a[i+1]*(self.x-self.x_prime[i])*q_1
            q_1 = q_2.copy()
            q_2 = q_temp.copy()
        
        if not np.any(np.isclose(q_2,0)):
            self.val = self.eval_not_const
            return
        
        else:
            raise ValueError('q_2 has a zero at a interpolation point, interpolant does not exist')

    def eval_const(self,z):
        return self.f_0

    def eval_not_const(self,z):
        fraction = 1.0
        for i in reversed(range(2,self.N)):
            fraction = 1+ self.a[i]*(z-self.x_prime[i-1])/fraction

        fraction = self.a[1]*(z-self.x_0)/fraction

        value = self.f_0 + fraction
        
        return value

class Greedy_Thiele_interpolator:
    #Implementation of Thiele interpolation (continued fraction interpolation)
    #using the greedy ordering of interpolation points described in 
    #arXiv:2109.10529 Oliver Salazar Celis

    def __init__(self,x,samples,rtol = 1e-13):
        #Input:
        #x: numpy vector of the interpolation points
        #samples: numpy vector of function values at interpolation points
        #rtol: relative tolerance for terimnation of interpolation procedure
        #Interpolation procedure is stopped if the maximum relatvie interpolation error 
        #at remaining points is below rtol

        self.M = x.shape[0]

class AAA:
    #Implementation of the AAA algorithm for rational interpolation
    #Original paper: The AAA Algorithm for Rational Approximation
    #Yuji Nakatsukasa, Olivier Sète, and Lloyd N. Trefethen
    #SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522

    def __init__(self,Z,samples,rtol = 1e-13,mmax = 100):
        #Input:
        #Z: numpy vector of interpolation points
        #samples: numpy vector of function values at interpolation points
        #rtol: relative tolerance, set to 1e-13 if omitted
        #mmax: maximum number of iterations set to min(len(z),mmax), default value 100
        
        self.M = Z.shape[0]
        SF = sp.dia_matrix((samples,0),(self.M,self.M),dtype = samples.dtype) #left scaling matrix
        J = np.array(range(self.M))
        R = np.mean(samples)
        z = np.array([],dtype = Z.dtype)
        f = np.array([],dtype = samples.dtype)
        #C = np.array([[]],dtype = Z.dtype)
        J = np.array(range(self.M))
        mmax = min(self.M,mmax)
        
        for i in range(mmax):
            j = np.argmax(np.abs(R-samples)) #find largest residual
            z = np.append(z,Z[j]) #add corresponding interpolation point to z
            f = np.append(f,samples[j]) #append corresponding value to f
            J = np.delete(J,J==j) #remove index from index array
            if i == 0:
                C = 1/(Z-Z[j])
                Sf = f
                A =  np.transpose(np.array([SF@C-C*Sf])) 
                s,w = (sl.svd(A[J,:],full_matrices = False)[1:]) #Get right singular vector with smalllest singular value
                w = np.conjugate(w[i,:])
                N = C*(w*f) #Numerator
                D = C*w #Denominator
                R = samples.copy() 
                R[J] = N[J]/D[J] #Rational approximation
                err = sl.norm(samples-R, ord = np.inf) #max error at interpolation points
                errvec = err 
            else:
                C = np.c_[C,1/(Z-Z[j])] #Add column to Cauchy matrix
                Sf = np.diag(f) #right scaling matrix
                #A = SF@C-C@Sf #Loewner matrix
                A = np.c_[A,(samples-f[-1])*C[:,-1]] #Loewner matrix
                s,w = (sl.svd(A[J,:],full_matrices = False)[1:]) #Get right singular vector with smalllest singular value
                w = np.conjugate(w[i,:])
                i0 = np.nonzero(w!=0)
                N = C[:,i0]@(w[i0]*f[i0]) #Numerator
                D = C[:,i0]@w[i0] #Denominator
                R = samples.copy()
                R[J] = N[J,0]/D[J,0] #Rational approximation
                err = sl.norm(samples-R, ord = np.inf) #max error at interpolation points
                errvec = np.c_[errvec,err]

            if err <= rtol*sl.norm(samples, ord = np.inf): #check convergence
                break
            
        #Keep arrays needed for evaluation
        self.z = z
        self.w = w
        self.f = f
        self.errvec = errvec #Save convergence history
        return

    def eval(self,zv):
        #Evaluate the interpolating funciton constructed with AAA algorithm
        C = np.zeros((zv.shape[0],self.z.shape[0]),dtype = zv.dtype)
        for index, z_i in np.ndenumerate(self.z): #Maybe which dimension to iterate over should depend on their sizes?
            C[:,index[0]] = 1/(zv-z_i)
        r = C@(self.w*self.f)/(C@self.w)
        nan_indices = np.argwhere(np.isnan(r))
        for index in nan_indices:
            r[index] = self.f[self.z == zv[index]]
        return r
        

def robust_pade(c,n,m = None,r_tol = 1e-14):
    #Implementation of algorithm copmuting 
    #Robust Padé approximants using SVD
    #Orginial paper: Robust Padé Approximation via SVD
    #Pedro Gonnet, Stefan Güttel and Lloyd N. Trefethen
    #SIAM Review, 55 1 (2013)

    if m is None:
        m = c.shape[0]-1-n

    if n + m +1 > c.shape[0]:
        raise ValueError('n+m is not less than degree of Taylor polynomial')
    
    #if 
    #zeros = np.zeros(m+n+1-c.shape[0],dtype = c.dtype)
    #print(c)
    #print(zeros)
    #if zeros.shape[0] != 0:
    #    c = np.concatenate(c,zeros)
    
    tau = r_tol*sl.norm(c)

    if sl.norm(c,np.inf)<tau:
        p = np.array([0.0])
        q = np.array([1.0])

        p = np.poly1d(np.flip(p))
        q = np.poly1d(np.flip(q))

        return p,q
    
    col = c
    row = np.insert(np.zeros(n,dtype=c.dtype),0,c[0])
    
    while True:

        if n ==0:
            if m==-1:
                p = c[0:1]
            else:
                p = c[:m+1]
            q = np.array([1.0])

            break

        

        Z = sl.toeplitz(col[:m+n+1],row[:n+1])

        #Get lower part of Toeplitz matrix for SVD.
        c_tilde = Z[m+1:,:]


        s = sl.svdvals(c_tilde)

        rho = np.sum(np.greater_equal(s,tau)) #Compute numerical rank of matrix
        print(np.sum(np.greater_equal(s,tau)))

        if rho == n:
            break

        else:
            m = m -(n-rho)
            n = rho



    if n>0:
        #Compute q from last right singular vector
        U,s,Vh = sl.svd(c_tilde)
        q = np.conjugate(Vh[-1,:])
        
        #compute p from upper part of Toeplitz matrix
        p = np.dot(Z[:m+1,:],q)

        #count leading zeros of q
        non_zero = np.where(np.abs(q)>=r_tol,1,0)
        lam = np.argmax(non_zero)

        #remove leading zeros
        if lam >0:
            q = q[lam:]
            p = p[lam:]


        #count trailing zeros of q
        non_zero = np.where(np.abs(q)>=r_tol,1,0)
        lam_trail = q.shape[0] - np.argmax(np.flip(non_zero))
        q = q[:lam_trail]


    #count trailing zeros of p
    non_zero = np.where(np.abs(p)>=tau,1,0)
    #print(p)

    if np.max(non_zero):
        lam_trail = p.shape[0] - np.argmax(np.flip(non_zero)) 
        p = p[:lam_trail]

    #Normalize Pade approximants
    p /= q[0]
    q /= q[0]

    #Return polynomial objects
    p = np.poly1d(np.flip(p))
    q = np.poly1d(np.flip(q))

    return p,q


def check_pade():
    e_exp = np.array([1.0, 1j*1.0, -1.0/2.0, -1j*1.0/6.0, 1.0/24.0, 1j*1.0/120.0])


    p,q = robust_pade(e_exp,2,1)

    print(p.coeffs)
    print(q.coeffs)

    print(f'e = {p(1)/q(1)}')

    p,q = pade(e_exp,2,1)

    print(p.coeffs)
    print(q.coeffs)

    print(f'e = {p(1)/q(1)}')

    return
    


    
