"""
[Linear Combination]
    Scaling two vectors with a basis [i, j] (two scalars i and j)
    and adding the vectors together is called linear combination.
    This operation is used in linear algebra to transform vectors.

    https://youtu.be/k7RM-ot2NWY?t=186

    For example, if you have two vectors v1 and v2,
    the linear combination of v1 and v2 with scalars i and j is:

        i * v1 + j * v2

    where i and j are scalars (or the basis of the axis).
    The result of the linear combination is another vector.
    The linear combination of two vectors is a linear transformation.

[FMA, fused multiply-add]
    CPU instruction that multiplies two numbers
    and adds a third number in a single instruction.

    https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation#Fused_multiply%E2%80%93add

    (a * b) + c

    The FMA function does both steps in one operation
    and rounds the result once, making it faster and more precise
"""

from typing import List
from math import fma
from lib import Vector


def linear_combination(
    vectors: List[Vector],  # List of vectors
    scalars: List[float]    # List of scalars (basis)
) -> List[float]:
    if len(vectors) != len(scalars):
        raise ValueError("Vectors and scalars must have the same size.")

    # Initialize the result vector with zeros
    result = Vector([0.0] * vectors[0].size())

    # Loop through each vector and scalar
    for vec, scl in zip(vectors, scalars):
        # Loop through each value in the vector
        for i in range(vec.size()):
            # Fused multiply-add (FMA) operation for each value
            # And add the previous vector computation (result[i]) to the result
            result[i] = fma(scl, vec[i], result[i])

    return result


def main():
    e1 = Vector([1., 0., 0.])
    e2 = Vector([0., 1., 0.])
    e3 = Vector([0., 0., 1.])

    v1 = Vector([1., 2., 3.])
    v2 = Vector([0., 10., -100.])

    print(linear_combination([e1, e2, e3], [10, -2, 0.5]))  # [10, -2, 0.5]
    print(linear_combination([v1, v2], [10, -2]))  # [10, 0, 230]


if __name__ == '__main__':
    main()
