#!/bin/sh

rm -rf logs *.log .tox .eggs public build dist
rm -rf docs/_build docs/AlgebraicNumber.* *.egg-info
find . -name "__pycache__" -exec rm -rf {} \;
find . -name "*.o" -exec rm -rf {} \;
find . -name "*.bak" -exec rm -rf {} \;

