"""
                    Linear Interpolation
In Mathematics
    -> https://fr.wikipedia.org/wiki/Interpolation_lin%C3%A9aire
    -> https://www.youtube.com/watch?v=M0R8-rYed0I

Linear Interpolation is the simplest method
to estimate the value taken by a continuous function between
two determined points.

It consists of using the affine function
(of the form f( x ) = a * x + b) passing through the two determined points

Suppose we know the values taken by a function f at two points Xa and Xb:
    f ( Xa ) = Ya
    f ( Xb ) = Yb

For X between Xa and Xb and we want to find Y value between Ya and Yb.
The formula of the affine function passing through these two points is:
    f ( X ) = Ya + (X - Xa) * ((Yb - Ya) / (Xb - Xa))
    Y       = Ya + (X - Xa) * ((Yb - Ya) / (Xb - Xa))

------------------------------------------------------------

[In Programming]

The linear interpolation function (lerp) takes three arguments:
    - u: scalar, vector, or matrix
    - v: scalar, vector, or matrix
    - t: scalar

    u as the starting point
    v as the ending point
    t as the interpolation factor (decimal between 0 and 1)

It returns the linear interpolation between u and v using the scalar t.
The formula used is:

    f(u, v, t) = u + t * (v - u)

For scalars, the function should return the lerp between two scalars.
For vectors, the function should return the lerp between two vectors.
For matrices, the function should return the lerp between two matrices.
"""

from typing import Union
from math import fma
from lib import Vector, Matrix


Scalar = Union[float, int, complex]


def lerp(
    u: Union[Scalar, Vector, Matrix],
    v: Union[Scalar, Vector, Matrix],
    t: float
) -> Union[Scalar, Vector, Matrix]:
    """
    Linear interpolation between u and v using the scalar t.
    Uses the formula: f(u, v, t) = u + t * (v - u).
    """
    # For scalars
    if isinstance(u, (float, int)) and isinstance(v, (float, int)):
        # Using fused multiply-add
        return fma(t, v - u, u)  # (t * (v - u)) + u

    # For vectors
    if isinstance(u, Vector) and isinstance(v, Vector):
        # Check if the vectors have the same size
        if u.size() != v.size():
            raise ValueError("Vectors must have the same size.")

        # Linear interpolation between vectors using FMA
        return Vector([fma(t, v[i] - u[i], u[i]) for i in range(u.size())])

    # For matrices
    if isinstance(u, Matrix) and isinstance(v, Matrix):
        # Check if the matrices have the same shape
        if u.shape() != v.shape():
            raise ValueError("Matrices must have the same shape.")
        rows, cols = u.shape()
        # Linear interpolation between matrices
        for i in range(rows):
            for j in range(cols):
                u[i, j] = fma(t, v[i, j] - u[i, j], u[i, j])
        return u

    raise ValueError("Invalid input types.")


def test_lerp():
    res = lerp(0.0, 1.0, 0.0)
    assert res == 0.0

    res = lerp(0.0, 1.0, 1.0)
    assert res == 1.0

    res = lerp(0.0, 1.0, 0.5)
    assert res == 0.5

    res = lerp(21.0, 42.0, 0.3)
    assert res == 27.3

    res = lerp(Vector([2.0, 1.0]), Vector([4.0, 2.0]), 0.3)
    assert res == Vector([2.6, 1.3])

    res = lerp(
        Matrix([[2.0, 1.0], [3.0, 4.0]]),
        Matrix([[20.0, 10.0], [30.0, 40.0]]),
        0.5
    )
    assert res == Matrix([[11.0, 5.5], [16.5, 22.0]])


def main():
    try:
        test_lerp()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
