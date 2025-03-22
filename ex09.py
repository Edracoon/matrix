"""
            Transpose of a Matrix

The transpose of a matrix AA is obtained by flipping
the matrix over its main diagonal (top-left to bottom-right).
This means:
    - Rows become columns.
    - Columns become rows.
n * m matrix will become an m * n matrix.

Example:
    A = [1, 2, 3]
        [4, 5, 6]

    A^T = [1, 4]
          [2, 5]
          [3, 6]

-> Why is the Transpose useful?

1. Swapping Rows and Columns
    If data is stored as rows but needs to be accessed as columns,
    transposing makes access more efficient.

2. Computing the Dot Product Using Matrices
    In linear algebra, matrix multiplication often involves the transpose.

3. Symmetric Matrices
    If A = A^T, the matrix is symmetric, which is important in engineering.

4. Orthogonality in Vectors
    The transpose is used in finding orthogonal vectors,
    which are useful in computer graphics and machine learning.
"""

from lib import Matrix


def test_transpose():
    print("--- Matrix trace ---")
    A = Matrix([[1, 2],
                [3, 4]])
    assert A.transpose() == Matrix([[1, 3],
                                    [2, 4]])

    A = Matrix([[1, 2, 3],
               [4, 5, 6]])
    assert A.transpose() == Matrix([[1, 4],
                                    [2, 5],
                                    [3, 6]])


def main():
    try:
        test_transpose()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
