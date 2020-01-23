# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""
A small set of functions for doing math operations.
"""

def add(arg1, arg2):
    """
    Function to add two input arguments
    """
    return arg1 + arg2

def mult(arg1, arg2):
    """
    Function to multiply two input arguments
    """
    return arg1 * arg2

def mod(arg1, arg2):
    """
    Function to return reminder from the division of two input arguments
    """
    return arg1 % arg2

def power(arg1, arg2):
    """
    Function to return first argument raised to the power of second argument
    """
    return arg1**arg2

def min(arg1, arg2):
    """
    Function to return minimum of the given arguments
    """
    if arg1 < arg2:
        minValue = arg1
    else:
        minValue = arg2 
    return minValue

def max(arg1, arg2):
    """
    Function to return maximum of the given arguments
    """
    if arg1 > arg2:
        maxValue = arg1
    else:
        maxValue = arg2 
    return maxValue
