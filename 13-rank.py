"""
    Rank of a Matrix

The rank of the matrix is the number of linearly
independent rows or columns in the matrix.

We can also say that this is number of dimensions
in the output of the linear transformation.

Example:
    A = [1, 2, 3]
        [4, 5, 6]
        [7, 8, 9]

    The rank of A is 3 because the matrix has 3 linearly independent rows.

Rank = 0	Matrix is all zeros
Rank = 1	All rows/columns lie on the same line
Rank = 2	Matrix rows/columns span a plane
Rank = n    Rows/columns span full space, matrix is invertible (full rank)

Algorithm:
    - Convert the matrix to row echelon form
    - Count the number of non-zero rows
    -> This count is the rank of the matrix

In Graphics/Physics:
 -> Rank reveals dimensionality: does your data live in a line / plane / 3D

In Machine Learning:
 -> Helps detect redundant data (e.g., duplicate or dependent features).
 -> Determines the number of parameters in a model.
 -> Helps with feature selection and dimensionality reduction.


"""

from lib import Matrix


def test_rank():
    print("--- Matrix rank ---")
    A = Matrix([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
    ])
    print("rank = ", A.rank())
    assert A.rank() == 2  # This matrix has rank 2 since row3 = 2*row2 - row1

    A = Matrix([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    print("rank = ", A.rank())
    assert A.rank() == 3  # Identity matrix has full rank

    A = Matrix([
        [1., 2., 3.],
        [2., 4., 6.],
        [3., 6., 9.]
    ])
    print("rank = ", A.rank())
    assert A.rank() == 1  # All rows are multiples of each other

    A = Matrix([
        [1., 2., 0., 0.],
        [2., 4., 0., 0.],
        [-1., 2., 1., 1.],
    ])
    print("rank = ", A.rank())
    assert A.rank() == 2  # All rows are linearly independent

    A = Matrix([
        [8., 5., -2.],
        [4., 7., 20.],
        [7., 6., 1.],
        [21., 18., 7.],
    ])
    print("rank = ", A.rank())
    assert A.rank() == 3  # All rows are linearly independent


def main():
    try:
        test_rank()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
