"""
            Trace of a matrix

The trace of a square matrix is the sum of its diagonal elements.

If A is an n by n matrix:

    trace(A) = a11 + a22 + ... + ann

Where Aii are the diagonal elements (where row index = column index).
The trace is only defined for square matrices.

Example:
    A = [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]

    trace(A) = 1 + 5 + 9 = 15

1. Used in Machine Learning and Physics

Invariance: The trace doesn’t change under certain transformations.
Quantum Mechanics: Used to describe quantum states.

2. Indicates Matrix Properties

If trace = 0, the matrix is traceless.
If trace is large, it gives information about the sum of its diagonal elements.
"""


from lib import Matrix


def test_trace():
    print("--- Matrix trace ---")
    A = Matrix([[1, 2],
                [3, 4]])
    assert A.trace() == 5

    A = Matrix([[1.0, 0.0],
                [0.0, 1.0]])
    assert A.trace() == 2.0

    A = Matrix([[3.0, -5.0],
                [6.0, 8.0]])
    assert A.trace() == 11.0

    A = Matrix([[-2., -8., 4.],
                [1., -23., 4.],
                [0., 6., 4.]])
    assert A.trace() == -21.0


def main():
    try:
        test_trace()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
