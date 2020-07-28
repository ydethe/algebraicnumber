import numpy as np
from numpy.polynomial import polynomial as poly

from AlgebraicNumber.utils import LogarithmicReverse, PolynomialReverse, PolynomialFromLogReverse


class AlgebraicNumber (object):
    def __init__(self, coeff, approx):
        self.approx = approx
        self.coeff = np.array(coeff, dtype=np.int64)

    def __mul__(self, b):
        '''

        Examples:
          >>> a = AlgebraicNumber([-2, 0, 1], approx=1.4)
          >>> b = AlgebraicNumber([-3, 0, 1], approx=1.7)
          >>> a*b

        '''
        la = LogarithmicReverse(self.coeff)
        lb = LogarithmicReverse(b.coeff)

        Da = len(la)-1
        Db = len(lb)-1

        la = np.pad(la, (0, Db), 'constant', constant_values=(0,0))
        lb = np.pad(lb, (0, Da), 'constant', constant_values=(0,0))
        
        lp = la*lb
        
        coeff = PolynomialFromLogReverse(lp)
        print(coeff)
        

if __name__ == '__main__':
    import doctest
    doctest.testmod()
