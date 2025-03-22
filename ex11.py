"""
The determinant of a Matrix is defined as a special number that is
defined only for square matrices (matrices that have the same number
of rows and columns). A determinant is used in many places in calculus
and other matrices related to algebra, it actually represents the matrix
in terms of a real number which can be used in solving a system of a
linear equation and finding the inverse of a matrix.

Geometric Meaning of the Determinant

The determinant represents the scaling factor
of the transformation described by the matrix :

    det(A) > 0 → The transfo preserves orientation (does not flip space).
    det(A) < 0 → The transfo flips space (mirror effect).
    det(A) = 0 → The transfo collapses space into a lower dimension.

A determinant equal to zero means that a matrix is a singular matrix.
A matrix is singular if it does not have an inverse, which means it
cannot be used to solve systems of linear equations.
"""
from lib import Matrix


def test_determinant():
    print("--- Matrix determinant ---")
    A = Matrix([
        [8., 5., -2., 4.],
        [4., 2.5, 20., 4.],
        [8., 5., 1., 4.],
        [28., -4., 17., 1.]
    ])
    print(A.determinant())
    assert A.determinant() == 1032

    A = Matrix([
        [1., -1.],
        [-1., 1.],
    ])

    print(A.determinant())
    assert A.determinant() == 0.0

    A = Matrix([
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.],
    ])

    print(A.determinant())
    assert A.determinant() == 8.0

    A = Matrix([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
    ])

    print(A.determinant())
    assert A.determinant() == -174.0


def main():
    try:
        test_determinant()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
