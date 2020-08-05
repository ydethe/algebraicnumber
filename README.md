# Algebre

[Page du dépôt git](https://gitlab.com/ydethe/algebraicnumber)

## Documentation

Pour générer la documentation du code, lancer :

    python setup.py doc

## Tests

Pour lancer les tests, choisissez une des options ci-dessous :

* `tox -e py`
* `python -m pytest --cov-report xml:test-results/coverage.xml --cov-config=coverage.cfg --cov AlgebraicNumber tests --junitxml=test-results/junit.xml --doctest-modules AlgebraicNumber`

Si tout va bien, vous devriez avoir la sortie suivante :

    ============================= test session starts =============================
    platform win32 -- Python 3.5.2, pytest-5.3.2, py-1.8.1, pluggy-0.13.1
    rootdir: .../AlgebraicNumber, inifile: setup.cfg
    plugins: cov-2.8.1
    collected 5 items
    
    tests\test_analyse.py ....                                               [ 80%]
    tests\test_util.py .                                                     [100%]
    
    - generated xml file: .../AlgebraicNumber/test-results/junit.xml
     -
    
    ----------- coverage: platform win32, python 3.5.2-final-0 -----------
    Coverage XML written to file test-results/coverage.xml
    
    ============================== 5 passed in 6.10s ==============================

## Rapport de couverture des tests

Une fois les tests lancés, le rapport de couverture des tests est disponible ici:

https://codecov.io/gl/ydethe/algebraicnumber

## Installation

Pour installer la librairie et les outils associés, lancer :

    python setup.py install --user

AlgebraicNumber met à disposition les outils suivants, accessibles en ligne de commande :

* AlgebraicNumber

