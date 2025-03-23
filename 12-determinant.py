"""
What is the inverse of a matrix?

    The inverse of a matrix A-1 is another matrix such that
    multiplying A by its inverse gives the identity matrix:

    A^-1 * A=In

    where:
        A is an n * n square matrix.
        In is the identity matrix of the same size.
        A^-1 exists only if A is invertible.

What does the inverse represent?

    -> Reversibility: If A represents a transfo, A^-1 reverses that transfo.
    -> Scaling Factor: Changes in volume/area due to AA are undone by A^-1.
    -> Solving Equations: It gives a formula for solving matrix equations.

Determinant Connection:

    A matrix is invertible if and only if det(A) ≠ 0.
    If det(A) = 0, the matrix is singular, meaning it has no inverse.

Rank Connection:

    A full-rank matrix (rank = n) is invertible.
    If the rank of A is less than n, it means its rows/columns are
    linearly dependent, making AA singular (non-invertible).
"""

from lib import Matrix


def test_inverse():
    print("--- Matrix inverse ---")
    A = Matrix([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    assert A.inverse() == Matrix([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    A = Matrix([
        [2., 0., 0.],
        [0., 2., 0.],
        [0., 0., 2.],
    ])
    assert A.inverse() == Matrix([
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ])

    A = Matrix([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
    ])
    print(A.inverse())
    assert A.inverse() == Matrix([
        [0.649425287, 0.097701149, -0.655172414],
        [-0.781609195, -0.126436782, 0.965517241],
        [0.143678160, 0.0747126436, -0.2068965517],
    ])


def main():
    try:
        test_inverse()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
