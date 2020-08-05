import math

import numpy as np
from scipy.optimize import root
from numpy.polynomial.polynomial import Polynomial
from scipy import linalg as lin
from numpy.polynomial import polynomial as P

from AlgebraicNumber.utils import cpslq

    
class AlgebraicNumber (object):
    @classmethod
    def unity(cls):
        return AlgebraicNumber([-1,1], 1.)
        
    @classmethod
    def zero(cls):
        return AlgebraicNumber([0,1], 0)
        
    @classmethod
    def constant(cls, a):
        b = -2*np.real(a)
        c = np.abs(a)**2
        return AlgebraicNumber([c,b,1], a)
        
    @classmethod
    def imaginary(cls):
        return AlgebraicNumber([1,0,1], 1j)
        
    def __init__(self, coeff, approx, _nosimp=False):
        self.coeff = coeff
        self.poly = Polynomial(coeff)
        self.approx = self.eval(approx)
        if not _nosimp:
            self._simplify()
        
    def _simplify(self):
        n = len(self.coeff)
        # cpslq([self.approx**k for k in range(n)])
        
        # print('self.coeff', self.coeff)
        # print('c', c)
        # Q,R = P.polydiv(self.coeff, c)
        # print('reste div', R)
        
        # self.coeff = c
        # self.poly = Polynomial(c)
        
    def eval(self, approx=None):
        if approx is None:
            approx = self.approx
            
        def fun(X):
            x,y = X
            z = x+1j*y
            P = self.poly(z)
            return [np.real(P),np.imag(P)]
            
        # print('x0', [np.real(approx),np.imag(approx)])
        sol = root(fun, x0 = [np.real(approx),np.imag(approx)])
        if sol.success:
            x,y = sol.x
            z = x+1j*y
            return z
        else:
            print(sol)
            raise ValueError
    
    def compagnon(self):
        n = len(self.coeff)-1
        res = np.zeros((n,n), dtype=np.int64)
        for i in range(n):
            res[i,-1] = -self.coeff[i]
            if i <= n-2:
                res[i+1,i] = self.coeff[-1]
            
        return res
    
    def inverse(self):
        coeff = self.coeff
        
        res = AlgebraicNumber(coeff[-1::-1], 1/self.approx)
        
        return res
        
    def __mul__(self, b):
        '''
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2*sqrt_3
        >>> # p.coeff
        # [-6, 0, 1]
        
        '''
        Ma = self.compagnon()
        if Ma.shape[0] > 1:
            Ka = Ma[1,0]
        else:
            Ka = 1
        
        Mb = b.compagnon()
        if Mb.shape[0] > 1:
            Kb = Mb[1,0]
        else:
            Kb = 1
        
        Mc = np.kron(Ma,Mb)
        
        P = np.poly(Mc)
        P = np.array([int(x) for x in np.round(P,0)[-1::-1]])
        n = len(P)
        Q = P*(Ka*Kb)**np.arange(n)
        
        res = AlgebraicNumber(Q, self.approx*b.approx)
        
        return res
        
    def __truediv__(self, b):
        '''
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2/sqrt_3
        >>> # p.coeff
        # [-2, 0, 3]
        
        '''
        ib = b.inverse()
        return self*ib
        
    def __neg__(self):
        '''
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> p = -sqrt_2
        >>> # p.coeff
        # [-2, 0, 1]
        
        '''
        n = len(self.coeff)
        coeff = np.array(self.coeff)
        
        R = coeff*(-1)**np.arange(n)
        
        res = AlgebraicNumber(list(R), -self.approx)
        
        return res
        
    def __sub__(self, b):
        nb = -b
        return self+nb
        
    def __add__(self, b):
        '''
        
        >>> sqrt_2 = AlgebraicNumber([-4,0,2], 1.4)
        >>> sqrt_3 = AlgebraicNumber([-9,0,3], 1.7)
        >>> p = sqrt_2+sqrt_3
        >>> # p.coeff
        # [1, 0, -10, 0, 1]
        >>> ref = np.sqrt(2) + np.sqrt(3)
        >>> np.abs(p.approx - ref) < 1e-10
        True
        
        '''
        Ma = self.compagnon()
        if Ma.shape[0] > 1:
            Ka = Ma[1,0]
        else:
            Ka = 1
        
        Mb = b.compagnon()
        if Mb.shape[0] > 1:
            Kb = Mb[1,0]
        else:
            Kb = 1
        
        I = np.eye(Ma.shape[0])
        J = np.eye(Mb.shape[0])
        
        Mc = np.kron(Ma,J)*Kb + np.kron(I,Mb)*Ka
        Kc = Ka*Kb
        
        P = np.poly(Mc)
        P = np.array([int(x) for x in np.round(P,0)[-1::-1]])
        n = len(P)
        Q = P*Kc**np.arange(n)
        
        res = AlgebraicNumber(Q, self.approx+b.approx)
        
        return res
        
    def conj(self):
        '''
        
        >>> z = AlgebraicNumber.unity() + AlgebraicNumber.imaginary()
        >>> z.coeff
        array([ 2, -2,  1])
        >>> p = z*z.conj()
        >>> # p.coeff
        # [-2, 1]
        
        '''
        coeff = self.coeff
        
        res = AlgebraicNumber(coeff, np.conj(self.approx))
        
        return res
        
        
if __name__ == '__main__':
    import doctest
    doctest.testmod()
    
    