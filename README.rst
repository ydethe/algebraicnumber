===============
AlgebraicNumber
===============


.. image:: https://img.shields.io/pypi/v/AlgebraicNumber.svg
        :target: https://pypi.python.org/pypi/AlgebraicNumber

.. image:: https://readthedocs.org/projects/algebraicnumber/badge/?version=latest
        :target: https://algebraicnumber.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://gitlab.com/ydethe/algebraicnumber/badges/master/pipeline.svg
   :target: https://gitlab.com/ydethe/algebraicnumber/pipelines

.. image:: https://codecov.io/gl/ydethe/algebraicnumber/branch/%5Cx6d6173746572/graph/badge.svg
  :target: https://codecov.io/gl/ydethe/algebraicnumber


A library to manipulate algebraic numbers


Documentation
-------------

Pour générer la documentation du code, lancer::

    python setup.py doc

Tests
-----

Pour lancer les tests, choisissez une des options ci-dessous::

* tox -e py
* python -m pytest --cov-report xml:test-results/coverage.xml --cov-config=coverage.cfg --cov AlgebraicNumber tests --junitxml=test-results/junit.xml --doctest-modules AlgebraicNumber

Si tout va bien, vous devriez avoir la sortie suivante::

    ============================= test session starts ==============================
    platform linux -- Python 3.5.2, pytest-6.0.1, py-1.9.0, pluggy-0.13.1
    cachedir: .tox/py35/.pytest_cache
    rootdir: /builds/ydethe/algebraicnumber, configfile: setup.cfg
    plugins: cov-2.10.0
    collected 31 items
    tests/test_ops.py ...........                                            [ 35%]
    AlgebraicNumber/AlgebraicNumber.py ......                                [ 54%]
    AlgebraicNumber/inria_utils.py ....                                      [ 67%]
    AlgebraicNumber/utils.py ..........                                      [100%]
    -- generated xml file: /builds/ydethe/algebraicnumber/test-results/junit.xml ---
    ----------- coverage: platform linux, python 3.5.2-final-0 -----------
    Name                                 Stmts   Miss  Cover
    --------------------------------------------------------
    AlgebraicNumber/AlgebraicNumber.py      97      6    94%
    AlgebraicNumber/__init__.py             13      0   100%
    AlgebraicNumber/inria_utils.py          91      3    97%
    AlgebraicNumber/utils.py               176     10    94%
    --------------------------------------------------------
    TOTAL                                  377     19    95%
    ============================== 31 passed in 0.87s ==============================

Rapport de couverture des tests
-------------------------------

Une fois les tests lancés, le rapport de couverture des tests est disponible ici:

https://codecov.io/gl/ydethe/algebraicnumber

Installation
------------

Pour installer la librairie et les outils associés, lancer::

    python setup.py install --user

