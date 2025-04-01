#!/usr/bin/env python3
"""
Run all tests for Tensorus.
"""

import os
import sys
import unittest

if __name__ == "__main__":
    # Add project root to path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    tests = loader.discover("tests")
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(tests)
    
    # Exit with appropriate status code
    sys.exit(not result.wasSuccessful()) 