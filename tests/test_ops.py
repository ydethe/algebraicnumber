import numpy as np
from numpy.polynomial import polynomial as P

from AlgebraicNumber.inria_utils import LogarithmicReverse, PolynomialReverse, PolynomialFromLogReverse

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestOperations (TestBase):
    def test_mul(self):
        a = [-2, 0, 1]
        b = [-3, 0, 1]
        
        Da = len(a)-1
        Db = len(b)-1
        D = Da+Db
        
        la = LogarithmicReverse(a, D=D, trim_res=False)
        lb = LogarithmicReverse(b, D=D, trim_res=False)
        
        lp = la*lb
        
        coeff = PolynomialFromLogReverse(lp, D=Da+Db)
        
        ref = np.array([36,0,-12,0,1])
        
        self.assertNpArrayAlmostEqual(coeff, ref, delta=1e-9)
        
        # print(P.polydiv([1],[0,1]))
        
    def test_log_reverse(self):
        a = [-2, 0, 1]
        D = 10
        la = LogarithmicReverse(a, D=D)
        ref = np.zeros(D+1)
        ref[::2] = 2**np.arange(1,2+(D+1)//2)
        self.assertNpArrayAlmostEqual(la, ref, delta=1e-9)
        
        h = np.array([1, 2, 3])
        lr = LogarithmicReverse(h)
        ref = np.array([ 2., -2/3, -2/9])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)
        
        lr = LogarithmicReverse(h, D=5)
        ref = np.array([ 2., -2/3, -2/9, 10/27, -14/81, -2/243])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)
        
        h = np.array([0, -3])
        lr = LogarithmicReverse(h)
        ref = np.array([1])
        self.assertNpArrayAlmostEqual(lr, ref, delta=1e-9)
        
    def test_polynomial_from_log_reverse(self):
        h = np.array([0, -3])
        lr = LogarithmicReverse(h)
        h2 = PolynomialFromLogReverse(lr, D=1)
        self.assertNpArrayAlmostEqual(h, h2*h[-1], delta=1e-9)
        
        for _ in range(10):
            n = np.random.randint(low=2, high=9)
            h = [0]
            while h[-1] == 0:
                h = np.random.randint(low=-5, high=5, size=n)
            
            lr = LogarithmicReverse(h)
            h2 = PolynomialFromLogReverse(lr, D=n-1)
            
            self.assertNpArrayAlmostEqual(h, h2*h[-1], delta=1e-9)
        
        
if __name__ == '__main__':
    import unittest
    unittest.main()
    
    