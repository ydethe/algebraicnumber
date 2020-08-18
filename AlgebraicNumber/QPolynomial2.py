from fractions import Fraction
from math import gcd, factorial

import numpy.polynomial.polynomial as P
import numpy as np


def nCr(n:int, p:int) -> int:
    return factorial(n)//(factorial(p)*factorial(n-p))
    

class QPolynomial (object):
    '''
    
    Examples:
      >>> p = QPolynomial()
      >>> p.printCoeff()
      '[]'
      >>> p = QPolynomial([1])
      >>> p.printCoeff()
      '[1]'
      >>> p = QPolynomial([-1,1])
      >>> p.printCoeff()
      '[-1,1]'
      >>> p = QPolynomial([-2,0,1])
      >>> p.printCoeff()
      '[-2,0,1]'
      >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
      >>> p.printCoeff()
      '[-2,0,2/3,0,1]'
      
    '''
    __slots__ = ['__coeff', 'F']
    def __init__(self, coeff:list=[], field=Fraction):
        self.F = field
        self.__coeff = []
        self.__simplify(coeff)
        
    def __simplify(self, coeff):
        n = len(coeff)
        ns = None
        for i in reversed(range(n)):
            if coeff[i] != 0 and ns is None:
                ns = i+1
                self.__coeff = [self.F(coeff[i])]
            elif not ns is None:
                self.__coeff = [self.F(coeff[i])] + self.__coeff
        
    def copy(self) -> 'QPolynomial':
        p = self.__coeff.copy()
        res = QPolynomial(coeff=p, field=self.F)
        return res

    def __len__(self) -> int:
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> len(p)
          5
          
        '''
        return len(self.__coeff)
        
    def __getitem__(self, i:int) -> 'F':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p[2]
          Fraction(2, 3)
          
        '''
        return self.__coeff[i]
        
    def getCoefficients(self, conv=None) -> 'F':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.getCoefficients()
          [Fraction(-2, 1), Fraction(0, 1), Fraction(2, 3), Fraction(0, 1), Fraction(1, 1)]
          >>> p.getCoefficients(conv=float)
          [-2.0, 0.0, 0.66666..., 0.0, 1.0]
          
        '''
        if conv is None:
            conv = self.F
        return [conv(x) for x in self.__coeff]
        
    def deg(self) -> int:
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.deg()
          4
          
        '''
        return len(self.__coeff) - 1
        
    def __repr__(self):
        sp = ''
        sc = ''
        
        if self.deg() < 0:
            return '\n0'
            
        for p,c in enumerate(self.getCoefficients()):
            if p == 0 and c < 0:
                x = '-' + str(-c) + ' '
                sc += x
                sp += ' '*len(x)
            elif p == 0 and c == 0:
                pass
            elif p == 0 and c == 1:
                if len(sc) == 0:
                    x = '1 '
                else:
                    x = ' '
                sc += x
                sp += ' '*len(x)
            elif p == 0 and c > 0:
                x = str(c) + ' '
                sc += x
                sp += ' '*len(x)
            elif p == 1 and c < 0:
                if len(sc) == 0:
                    x = '-' + str(-c) + '.X '
                else:
                    x = '- ' + str(-c) + '.X '
                sc += x
                sp += ' '*len(x)
            elif p == 1 and c == 0:
                pass
            elif p == 1 and c == 1:
                if len(sc) == 0:
                    x = 'X '
                else:
                    x = '+ X '
                sc += x
                sp += ' '*len(x)
            elif p == 1 and c > 0:
                if len(sc) == 0:
                    x = str(c) + '.X '
                else:
                    x = '+ ' + str(c) + '.X '
                sc += x
                sp += ' '*len(x)
            elif p > 1 and c < 0:
                y = str(p)
                if len(sc) == 0:
                    x = '-' + str(-c) + '.X'
                else:
                    x = '- ' + str(-c) + '.X'
                sc += x
                sp += ' '*len(x)
                sc +=  ' '*(len(y)+1)
                sp += y + ' '
            elif p > 1 and c == 0:
                pass
            elif p > 1 and c == 1:
                y = str(p)
                if len(sc) == 0:
                    x = 'X'
                else:
                    x = '+ X'
                sc += x
                sp += ' '*len(x)
                sc +=  ' '*(len(y)+1)
                sp += y + ' '
            elif p > 1 and c > 0:
                y = str(p)
                if len(sc) == 0:
                    x = str(c) + '.X'
                else:
                    x = '+ ' + str(c) + '.X'
                sc += x
                sp += ' '*len(x)
                sc +=  ' '*(len(y)+1)
                sp += y + ' '
                
        return sp + '\n' + sc
    
    def printCoeff(self) -> str:
        res = [str(x) for x in self.getCoefficients()]
        return '[' + ','.join(res)+']'
        
    def __call__(self, x) -> float:
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p(0)
          -2.0...
          
        '''
        return P.polyval(x, self.getCoefficients(conv=float))
    
    def __neg__(self) -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> (-p).printCoeff()
          '[2,0,-2/3,0,-1]'
          
        '''
        res = []
        n = len(self)
        
        for i in range(n):
            r = -self[i]
            res.append(r)
            
        return QPolynomial(res, field=self.F)
        
    def __add__(self, b:'QPolynomial') -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p+q).printCoeff()
          '[-4,2/5,5/3,0,1]'
          
        '''
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)
            
        res = []
        n = len(self)
        p = len(b)
        
        for i in range(max(n,p)):
            if i < n and i < p:
                r = self[i] + b[i]
            elif i < n and i >= p:
                r = self[i]
            elif i >= n and i < p:
                r = b[i]
            else:
                r = self.F(0)
            res.append(r)
            
        return QPolynomial(res, field=self.F)
        
    def __sub__(self, b:'QPolynomial') -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p-q).printCoeff()
          '[0,-2/5,-1/3,0,1]'
          >>> (p-p).printCoeff()
          '[]'
          
        '''
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)
            
        mb = -b
        return self + mb
        
    def __truediv__(self, b:'QPolynomial') -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> (p/2).printCoeff()
          '[-1,0,1/3,0,1/2]'
          
        '''
        res = []
        n = len(self)
        
        for i in range(n):
            r = self[i]/self.F(b)
            res.append(r)
            
        return QPolynomial(res, field=self.F)
        
    def __mul__(self, b:'QPolynomial') -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> (p*q).printCoeff()
          '[4,-4/5,-10/3,4/15,-4/3,2/5,1]'
          >>> (p*3).printCoeff()
          '[-6,0,2,0,3]'
          
        '''
        if not isinstance(b, QPolynomial):
            b = QPolynomial(coeff=[b], field=self.F)
            
        res = []
        n = len(self)
        p = len(b)
        
        for i in range(n+p+1):
            r = self.F(0)
            for k in range(i+1):
                j = i-k
                if k < n and j < p and j >= 0:
                    r += self[k]*b[j]
            res.append(r)
            
        return QPolynomial(res, field=self.F)
    
    def termWiseMul(self, b:'QPolynomial') -> 'QPolynomial':
        '''Hadamard product of the polynomials
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> p.termWiseMul(q).printCoeff()
          '[4,0,2/3]'
          
        '''
        da = self.deg()
        db = b.deg()
        d = min(da,db)
        
        c = [self[i]*b[i] for i in range(d+1)]
        
        return QPolynomial(coeff = c, field=self.F)
    
    def termWiseDiv(self, b:'QPolynomial') -> 'QPolynomial':
        '''Hadamard product of the polynomials
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> q = QPolynomial([-2,Fraction(2,5),1,0,0])
          >>> p.termWiseDiv(q).printCoeff()
          '[1,0,2/3]'
          
        '''
        da = self.deg()
        db = b.deg()
        d = min(da,db)
        
        c = [self[i]/b[i] for i in range(d+1)]
        
        return QPolynomial(coeff = c, field=self.F)
        
    def isNull(self) -> bool:
        '''Checks if a polynomial is null
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.isNull()
          False
          >>> (p-p).isNull()
          True
          >>> p = QPolynomial()
          >>> p.isNull()
          True
        
        '''
        return len(self) == 0
        
    def __eq__(self, b:'QPolynomial') -> bool:
        p = self - b
        return p.isNull()
        
    def __neq__(self, b:'QPolynomial') -> bool:
        return not self.__eq__(b)
        
    def integrate(self) -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.integrate().printCoeff()
          '[0,-2,0,2/9,0,1/5]'
          
        '''
        res = [self.F(0)]
        n = len(self)
        
        for i in range(n):
            r = self[i]/self.F(i+1)
            res.append(r)
            
        return QPolynomial(res, field=self.F)
        
    def derive(self) -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,Fraction(2,3),0,1,0])
          >>> p.derive().printCoeff()
          '[0,4/3,0,4]'
          
        '''
        res = []
        n = len(self)
        
        for i in range(1,n):
            r = self[i]*self.F(i)
            res.append(r)
            
        return QPolynomial(res, field=self.F)
        
    def roots(self) -> np.array:
        '''
        
        Examples:
          >>> p = QPolynomial([-2,0,1])
          >>> p.roots()
          array([-1.4142135...,  1.4142135...]...)
          
        '''
        return P.polyroots(self.getCoefficients(conv=float))
        
    def squareFreeFact(self):
        r"""
        
        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
          >>> R.squareFreeFact().printCoeff()
          '[-1,0,0,0,1]'
          
        """
        n = self.deg()
        p = self.copy()
        
        while True:
            dp = p.derive()
            g = Qpolygcd(p, dp)
            if g == QPolynomial(coeff=[1], field=self.F):
                return p
                
            p, r = Qpolydiv(p, g)
            if not r.isNull():
                raise AssertionError(r)
                
    def compose(self, S:'QPolynomial') -> 'QPolynomial':
        r"""Computes self(S(X))
        
        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 1])
          >>> S = QPolynomial(coeff=[-4, 5])
          >>> R.compose(S).printCoeff()
          '[21,-45,25]'
          
        """
        res = QPolynomial(field=self.F)
        sp = QPolynomial(coeff=[self.F(1)], field=self.F)
        for rk in self:
            if rk != 0:
                res = res + sp*rk
            sp = sp * S

        return res
        
    def translate(self, a:'QPolynomial') -> 'QPolynomial':
        r"""
        
        Examples:
          >>> R = QPolynomial(coeff=[0, 0, 1])
          >>> R.translate(2).printCoeff()
          '[4,4,1]'
          
        """
        n = self.deg()

        q = (n + 1)*[None]
        for k in range(n + 1):
            q[k] = self.F(0)
            for p in range(k, n + 1):
                q[k] += nCr(p, k) * self[p] * a ** (p - k)

        return QPolynomial(coeff=q, field=self.F)
    
    def truncate(self, deg:int) -> 'QPolynomial':
        """
        
        Examples:
          >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
          >>> R.truncate(2).printCoeff()
          '[1,-1]'
          
        """
        p = self.getCoefficients()[:deg+1]
        
        return QPolynomial(coeff=p, field=self.F)

    def reverse(self) -> 'QPolynomial':
        '''
        
        Examples:
          >>> p = QPolynomial([0,-2,0,1])
          >>> p.reverse().printCoeff()
          '[1,0,-2]'
          
        '''
        return QPolynomial(self.getCoefficients()[-1::-1], field=self.F)
        
        
def Qpolydiv(n, d):
    '''
    
    Examples:
      >>> n = QPolynomial(coeff=[-4, 0, -2, 1])
      >>> d = QPolynomial(coeff=[-3, 1])
      >>> q,r = Qpolydiv(n, d)
      >>> q.printCoeff()
      '[3,1,1]'
      >>> r.printCoeff()
      '[5]'
  
    '''
    F = n.F
    
    if d.isNull():
        raise ZeroDivisionError
    
    q = QPolynomial(field=F)
    r = n.copy() # At each step n = d × q + r
    
    while not r.isNull() and r.deg() >= d.deg():
        c = r[-1] / d[-1] # Divide the leading terms
        t = QPolynomial(coeff=(r.deg()-d.deg())*[F(0)] + [c], field=F)
        q = q + t
        r = r - t*d
        
    return (q, r)
    
    
def Qpolygcd(a, b):
    r"""
    
    Examples:
      >>> R = QPolynomial(coeff=[1, -1, 0, 0, -1, 1])
      >>> S = QPolynomial(coeff=[-1, 0, 0, -4, 5])
      >>> g = Qpolygcd(S, R)
      >>> g.printCoeff()
      '[-1,1]'
      
    """
    m = a.deg()
    n = b.deg()
    
    if n > m:
        a, b = b, a
        n, m = m, n

    # Here, deg(a) >= deb(b)
    while True:
        q, r = Qpolydiv(a, b)
        
        a, b = b, r
        
        if b.isNull():
            return a / a[-1]
            
            
def Qnpolymul(*polynomials):
    lp = list(polynomials)
    
    if len(lp) == 0:
        return QPolynomial(coeff=[1])
        
    F = lp[0].F
    res = QPolynomial(coeff=[F(1)], field=F)
    for q in lp:
        res = res * q
        
    return res


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    
    