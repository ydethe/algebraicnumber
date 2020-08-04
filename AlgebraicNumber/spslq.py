"""
Implements the PSLQ algorithm for integer relation detection,
and derivative algorithms for constant recognition.
"""

import numpy as np
import scipy.linalg as lin
from numpy.linalg import matrix_rank


def proj(dir, vec):
    '''
    
    Projection de vec sur dir
    
    '''
    return np.dot(dir,vec)/np.dot(dir,dir)*dir
    
def hyperplane_matrix(x, tol=1e-9):
    '''
    
    x is a matrix in M(n,t) of t vectors in R^n
    The rank of x must be t
    
    '''
    n,t = x.shape
    
    assert(matrix_rank(x) == t)
    
    x2 = np.empty((n,t))
    x2[:,0] = x[:,0]/lin.norm(x[:,0])
    for k in range(1,t):
        x2[:,k] = x[:,k]
        for j in range(k):
            x2[:,k] -= proj(dir=x2[:,j], vec=x[:,k])
        x2[:,k] /= lin.norm(x2[:,k])
        
    b2 = np.zeros((n,n))
    b2[0,0] = 1
    b = np.zeros(n)
    b[0] = 1
    for j in range(t):
        b2[:,0] -= proj(dir=x2[:,j], vec=b)
    b2[:,0] = b2[:,0]/lin.norm(b2[:,0])
    b[0] = 0
    
    inz = [0]
    for i in range(1,n):
        b[i] = 1
        b2[i,i] = 1
        for j in range(t):
            b2[:,i] -= proj(dir=x2[:,j], vec=b)
        for j in range(i):
            nb2 = np.dot(b2[:,j],b2[:,j])
            if nb2 > tol**2:
                b2[:,i] -= proj(dir=b2[:,j], vec=b)
        
        nb2 = np.dot(b2[:,i],b2[:,i])
        if nb2 > tol**2:
            b2[:,i] /= np.sqrt(nb2)
            inz.append(i)
        else:
            b2[:,i] = 0.
        b[i] = 0
        
    H = b2[:,inz]
    
    return x2, b2, H
    
def generalized_hermite_reduction(H_in, tol=1e-9):
    H = H_in.copy()
    
    n,p = H_in.shape
    t = n-p
    
    D = np.eye(n, dtype=np.int64)
    
    for i in range(n-1,n):
        for j in reversed(range(n-t)):
            if np.abs(H_in[j,j]) < tol:
                print(H_in)
                raise ValueError
                
            q = nint(H_in[i,j]/H_in[j,j])
            
            for k in range(n):
                D[i,k] = D[i,k] - q*D[j,k]
            
            H = D@H_in
            
            for s1 in range(n-t,n-1):
                for s2 in range(s1+1,n):
                    if np.abs(H[s1,n-t-1]) < tol and np.abs(H[s2,n-t-1]) > tol:
                        D[[s1,s2],:] = D[[s2,s1],:]
    
    assert(np.abs(lin.det(D)) == 1)
    
    return D
    
def nint(x):
    return np.int64(np.floor(x+0.5))

def spslq(x, tol=1e-10, maxcoeff=1000, maxsteps=100, verbose=False):
    n,t = x.shape
    x2, b2, H = hyperplane_matrix(x, tol)
    
    B = np.eye(n)
    
    D = generalized_hermite_reduction(H, tol=tol)
    Di = lin.inv(D)
    x = Di.T@x
    H = D@H
    B = B@Di
    
    gam = np.array([np.sqrt(6/3)**k for k in range(1,n-t+1)])
    while True:
        yy = np.array([gam[k]*np.abs(H[k,k]) for k in range(n-t-1)])
        r = np.argmax(yy)
        
        a = H[r,r]
        b = H[r+1,r]
        l = H[r+1,r+1]
        d = np.sqrt(b**2 + l**2)
        
        x[r],x[r+1] = x[r+1],x[r]
        H[[r,r+1],:] = H[[r+1,r],:]
        B[:,[r,r+1]] = B[:,[r+1,r]]
        
        Q = np.eye(n-t)
        Q[r,r] = b/d; Q[r,r+1] = -l/d
        Q[r+1,r] = l/d; Q[r+1,r+1] = b/d
        H = H@Q
        
        D = generalized_hermite_reduction(H)
        Di = lin.inv(D)
        x = Di.T@x
        H = D@H
        B = B@Di
        
        yy = np.array([np.abs(H[k,k]) for k in range(n-t)])
        G = 1/np.max(yy)
        
        # xx = np.array([lin.norm(x[:,k]) for k in range(t)])
        # i0 = np.argmin(xx)
        # if xx[i0] < tol:
            # return B[:,i0]
        
        i0 = np.argmin(yy)
        # print(G, yy[i0], B[:,i0])
        if yy[i0] < tol:
            return B[:,i0]
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    