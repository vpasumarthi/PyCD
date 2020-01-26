"""
Unit and regression test for the PyCD package.
"""

# Import package, test suite, and other packages as needed
import PyCD
import pytest
import sys

def test_PyCD_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "PyCD" in sys.modules
