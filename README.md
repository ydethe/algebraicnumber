# AlgebraicNumber

[![image](https://img.shields.io/pypi/v/AlgebraicNumber.svg)](https://pypi.python.org/pypi/AlgebraicNumber)
[![image](https://gitlab.com/ydethe/algebraicnumber/badges/master/pipeline.svg)](https://gitlab.com/ydethe/algebraicnumber/pipelines)
[![codecov](https://codecov.io/gl/ydethe/algebraicnumber/branch/master/graph/badge.svg?token=T84J2LCHND)](https://codecov.io/gl/ydethe/algebraicnumber)

A library to manipulate algebraic numbers

## Documentation

To generate the documentation,run:

    nox
    
https://ydethe.gitlab.io/algebraicnumber/docs

## Usage

    >>> z = AlgebraicNumber.unity() + AlgebraicNumber.imaginary()
    >>> z.poly.printCoeff()
    '[2,-2,1]'
    >>> p = z*z.conj()
    >>> p.poly.printCoeff()
    '[-2,1]'
