#!/usr/bin/env python
"""
testing the math.py module.
"""

import PyCT as pyct
import pytest

def test_add():
    assert pyct.math.add(3, 4) == 7
    assert pyct.math.add(4, 4) == 8

testdata = [
    (2, 5, 10),
    (1, 2, 2),
    (3, 4, 12),
    (3, 3, 9),
    (6, 7, 42)
]
@pytest.mark.parametrize("a,b,expected", testdata)
def test_mult(a, b, expected):
    assert pyct.math.mult(a, b) == expected
    assert pyct.mult(b, a) == expected

def test_mod():
    assert pyct.math.mod(4, 3) == 1
    assert pyct.math.mod(6, 3) == 0

def test_power():
    assert pyct.math.power(4, 2) == 16
    assert pyct.power(2, 3) == 8

def test_min():
    assert pyct.math.min(2, 3) == 2
    assert pyct.math.min(3, 3) == 3

def test_max():
    assert pyct.math.max(3, 4) == 4
    assert pyct.math.max(4, 4) == 4
