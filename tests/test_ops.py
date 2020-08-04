import numpy as np
from numpy.polynomial import polynomial as poly

from AlgebraicNumber.inria_utils import LogarithmicReverse, PolynomialReverse, PolynomialFromLogReverse

try:
    from tests.TestBase import TestBase
except Exception as e:
    from TestBase import TestBase


class TestOperations (TestBase):
    def test_mul(self):
        a = [-2, 0, 1]
        b = [-3, 0, 1]
        
        la = LogarithmicReverse(a)
        lb = LogarithmicReverse(b)
        
        Da = len(la)-1
        Db = len(lb)-1

        la = np.pad(la, (0, Db), 'constant', constant_values=(0,0))
        lb = np.pad(lb, (0, Da), 'constant', constant_values=(0,0))
        
        lp = la*lb
        
        coeff = PolynomialFromLogReverse(lp)
        print(coeff)
        
      
if __name__ == '__main__':
    unittest.main()
    
    