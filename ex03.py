"""
- What is the Dot Product Between Two Vectors?

    The dot product is a way to multiply two vectors
    and get a single number (a scalar) as the result.

    1. Geometric Interpretation: Projection

    The dot product tells you how much one vector points in the direction
    of another vector. It's related to the angle between the two vectors.

    2. Algebraic Interpretation: Sum of Products

    Dot product is the sum of the products of the corresponding elements
    of the two vectors. It's a way to multiply vectors element-wise.

- What Does the Dot Product Measure?

The dot product measures the alignment between two vectors:

    Large positive value: The vectors point in similar directions.
    Zero: The vectors are perpendicular.
    Large negative value: The vectors point in opposite directions.
"""

from lib import Vector


def test_dot():
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert v1.dot(v2) == 32

    v1 = Vector([2, 2])
    v2 = Vector([2, 2])
    assert v1.dot(v2) == 8

    # Two vectors pointing in opposite directions
    v1 = Vector([1, 1])
    v2 = Vector([-1, -1])
    assert v1.dot(v2) == -2

    # Two vectors pointing in perpendicular directions
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    assert v1.dot(v2) == 0

    # Test for vectors with different sizes
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6, 7])
    try:
        v1.dot(v2)
        assert False
    except AssertionError:
        assert True


if __name__ == "__main__":
    try:
        test_dot()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")
