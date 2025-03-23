"""
1-Norm (Taxicab or Manhattan Norm)
    This is the sum of the absolute values of all elements in the vector.

    What does it represent?
    - It measures the distance you would travel if you could only move
    along a grid (like streets in a city).
    - It's called the Manhattan distance because it represents
    how a taxi would drive through a city grid.

    Formula:
        ||v|| = |v1| + |v2| + ... + |vn|

    Example:
        v = [1, 2, 3]
        ||v|| = |1| + |2| + |3| = 6

2-Norm (Euclidean Norm)
    The square root of the sum of the squares of all elems in the vector.

    What does it represent?
    - It measures the straight-line distance between two points.
    - It's called the Euclidean distance because it represents
    the shortest distance between two points in Euclidean space.

    Formula:
        ||v|| = sqrt(v1^2 + v2^2 + ... + vn^2)

    Example:
        v = [1, 2, 3]
        ||v|| = sqrt(1^2 + 2^2 + 3^2) = 3.7416573867739413

Infinity-Norm (Maximum Norm)
    The maximum absolute value of all elements in the vector.

    What does it represent?
    - It measures the largest single value in the vector.
    - Useful when you want to know the most significant component of a vector.

    Formula:
        ||v|| = max(|v1|, |v2|, ..., |vn|)

    Example:
        v = [1, 2, 3]
        ||v|| = max(|1|, |2|, |3|) = 3

/!/ Norms always return real numbers, even for complex-valued vectors.
"""

from lib import Vector


def test_all_norms():
    u = Vector([0, 0, 0])
    assert u.norm_1() == 0.0
    assert u.norm() == 0.0
    assert u.norm_inf() == 0.0
    # 0.0, 0.0, 0.0

    u = Vector([1, 2, 3])
    assert u.norm_1() == 6.0
    assert u.norm() >= 3.741
    assert u.norm_inf() == 3.0
    # 6.0, 3.74165738, 3.0

    u = Vector([-1, -2, -3])
    assert u.norm_1() == 6.0
    assert u.norm() >= 3.741
    assert u.norm_inf() == 3.0
    # 6.0, 3.74165738, 3.0

    u = Vector([-1, -2])
    assert u.norm_1() == 3.0
    assert u.norm() >= 2.236
    assert u.norm_inf() == 2.0
    # 3.0, 2.23606798, 2.0

    u = Vector([1, 0, -8])
    assert u.norm_1() == 9.0
    assert u.norm() >= 8.062
    assert u.norm_inf() == 8.0
    # 9.0, 8.06225775, 8.0


def main():
    try:
        test_all_norms()
        print("All tests passed")
    except AssertionError:
        print("Some tests failed")


if __name__ == "__main__":
    main()
