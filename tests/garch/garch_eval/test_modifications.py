#!/usr/bin/env python3
"""Test script to verify garch_eval modifications work correctly."""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")

    # Test main module imports
    try:
        import src.garch.garch_eval.main  # noqa: F401

        print("✓ Main module imports successful")
    except ImportError as e:
        print(f"✗ Main module import failed: {e}")
        raise AssertionError(f"Main module import failed: {e}") from e

    # Test specific function imports
    try:
        import src.garch.garch_eval.metrics  # noqa: F401

        print("✓ Function imports successful")
    except ImportError as e:
        print(f"✗ Function import failed: {e}")
        raise AssertionError(f"Function import failed: {e}") from e


def test_function_signatures():
    """Test that function signatures are correct."""
    print("Testing function signatures...")

    import inspect
    from src.garch.garch_eval.metrics import compute_all_metrics

    # Check main function signature
    sig = inspect.signature(compute_all_metrics)
    params = list(sig.parameters.keys())

    expected_params = [
        "e_test",
        "s2_test",
        "dist",
        "nu",
        "alphas",
        "lambda_skew",
        "use_mz_calibration",
    ]
    assert (
        params == expected_params
    ), f"compute_all_metrics signature incorrect. Expected: {expected_params}, Got: {params}"
    print("✓ compute_all_metrics signature correct")


def test_basic_functionality():
    """Test basic functionality of refactored functions."""
    print("Testing basic functionality...")

    import numpy as np
    from src.garch.garch_eval.metrics import _prepare_mz_calibration

    # Test _prepare_mz_calibration
    e_test = np.array([0.1, -0.05, 0.02])
    s2_test = np.array([1.0, 0.8, 1.2])

    try:
        result = _prepare_mz_calibration(e_test, s2_test, False)
        assert (
            len(result) == 4
        ), f"_prepare_mz_calibration returns incorrect tuple length: {len(result)}"
        print("✓ _prepare_mz_calibration returns correct tuple length")
    except Exception as e:
        print(f"✗ _prepare_mz_calibration failed: {e}")
        raise AssertionError(f"_prepare_mz_calibration failed: {e}") from e


def main():
    """Run all tests."""
    print("=" * 50)
    print("Testing garch_eval modifications")
    print("=" * 50)

    tests = [
        test_imports,
        test_function_signatures,
        test_basic_functionality,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()

    print("=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
