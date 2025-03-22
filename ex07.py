"""
                Linear Maps

In finite-dimensional vector spaces, every linear map can be
represented as a matrix multiplied by a vector.

    T(v)=A⋅v
    T(v)=A⋅v

Where:

    T is the linear map (the transformation).
    A is a matrix representing the transformation.
    v is the vector being transformed.
    A⋅v is the resulting transformed vector.

A linear map can:

    - Rotate a vector (like turning an arrow).
    - Scale (stretch or shrink) a vector.
    - Reflect a vector over an axis or plane.
    - Shear a shape (like slanting a rectangle into a parallelogram).


                Matrix-by-Matrix Multiplication

Matrix-by-matrix multiplication is a way to combine two linear transformations
(represented as matrices) into a single transformation. This operation is
fundamental in linear algebra, computer graphics, physics, and machine learning

When we multiply two matrices:
    C = A ⋅ B
    C = A ⋅ B

    A is an m by n matrix.
    B is an n by p matrix.

    The result C is an m by p matrix.

What Does Matrix Multiplication Represent?

Matrix multiplication represents applying one transformation after another
    A applies the first transformation.
    B applies the second transformation.
    The result C is a combined transformation that does both at once.
For example:
    Rotation followed by scaling.
    Shearing followed by reflection.

A=[[1, 3], [2, 4]]
B=[[5, 7], [6, 8]]

c11 = a11 * b11 + a12 * b21
c12 = a11 * b12 + a12 * b22
c21 = a21 * b11 + a22 * b21
c22 = a21 * b12 + a22 * b22

C = A ⋅ B = [[1*5 + 2*6, 1*7 + 2*8], [3*5 + 4*6, 3*7 + 4*8]]
"""

from lib import Vector, Matrix


def test_mul_vec():
    print("--- Vector-matrix multiplication ---")
    M = Matrix([[1.0, 0.0],
                [0.0, 1.0]])
    V = Vector([4.0, 2.0])
    assert M.mul_vec(V) == Vector([4.0, 2.0])

    M = Matrix([[2.0, 0.0],
                [0.0, 2.0]])
    V = Vector([4.0, 2.0])
    assert M.mul_vec(V) == Vector([8.0, 4.0])

    M = Matrix([[2.0, -2.0],
                [-2.0, 2.0]])
    V = Vector([4.0, 2.0])
    assert M.mul_vec(V) == Vector([4.0, -4.0])

    M = Matrix([[1.0, 2.0],
                [3.0, 4.0]])
    V = Vector([5.0, 6.0])
    assert M.mul_vec(V) == Vector([17.0, 39.0])


def test_mul_mat():
    print("--- Matrix multiplication ---")
    A = Matrix([[1, 2],
                [3, 4]])
    B = Matrix([[5, 6],
                [7, 8]])
    assert A.mul_mat(B) == Matrix([[19.0, 22.0], [43.0, 50.0]])

    A = Matrix([[1.0, 0.0],
                [0.0, 1.0]])
    B = Matrix([[1.0, 0.0],
                [0.0, 1.0]])
    assert A.mul_mat(B) == Matrix([[1.0, 0.0], [0.0, 1.0]])

    A = Matrix([[3.0, -5.0],
                [6.0, 8.0]])
    B = Matrix([[2.0, 1.0],
                [4.0, 2.0]])
    assert A.mul_mat(B) == Matrix([[-14.0, -7.0], [44.0, 22.0]])


def main():
    try:
        test_mul_vec()
        print("test_mul_vec() tests passed.")

        test_mul_mat()
        print("test_mul_mat() tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
