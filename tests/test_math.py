#!/usr/bin/env python
"""
testing the math.py module.
"""

import KineticModel as km
import pytest

def test_add():
    assert km.math.add(3, 4) == 7
    assert km.math.add(4, 4) == 8

testdata = [
    (2, 5, 10),
    (1, 2, 2),
    (3, 4, 12),
    (3, 3, 9),
    (6, 7, 42)
]
@pytest.mark.parametrize("a,b,expected", testdata)
def test_mult(a, b, expected):
    assert km.math.mult(a, b) == expected
    assert km.mult(b, a) == expected

def test_mod():
    assert km.math.mod(4, 3) == 1
    assert km.math.mod(6, 3) == 0

def test_power():
    assert km.math.power(4, 2) == 16
    assert km.power(2, 3) == 8

def test_min():
    assert km.math.min(2, 3) == 2
    assert km.math.min(3, 3) == 3

def test_max():
    assert km.math.max(3, 4) == 4
assert km.math.max(4, 4) == 4
