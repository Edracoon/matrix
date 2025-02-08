"""
                Cross Product

The cross product of two 3D vectors is a new 3D vector that is perpendicular
to both original vectors. It is often used in physics, engineering,
and computer graphics to determine orientations and normal vectors.

            Mathematical Definition

For two vectors U and V in a 3D space:
    U = (Ux, Uy, Uz)  V=(Vx, Vy, Vz)
    U = (Ux, Uy, Uz)  V=(Vx, Vy, Vz)

The cross product W is defined as:
    W = U x V = [Wx Wy Wz]

    Where:
        Wx=(UyVz - UzVy)
        Wy=(UzVx - UxVz)
        Wz=(UxVy - UyVx)

    Or in matrix form:
        W = U x V = | i  j  k  |
                    | Ux Uy Uz |
                    | Vx Vy Vz |

    Where i, j, and k are the unit vectors along the x, y, and z axes.

What Does the Cross Product Represent?
    1. A vector that is perpendicular to both input vectors
    2. Direction given by the right-hand rule
    3. Magnitude represents the area of the parallelogram formed

Use Cases:
    - Calculating the normal vector of a plane
    - Determining the orientation of a polygon
    - Calculating the torque in physics
    - Generating random vectors for 3D graphics
"""


from lib import Vector
from math import fma


def cross(u: "Vector", v: "Vector") -> "Vector":
    """Computes the cross product of two 3D vectors."""
    if u.size() != 3 or v.size() != 3:
        raise ValueError("Cross product is only defined for 3D vectors.")
    Ux, Uy, Uz = u.values
    Vx, Vy, Vz = v.values

    return Vector([
        fma(Uy, Vz, -fma(Uz, Vy, 0)),  # w_x = (Uy * Vz - Uz * Vy)
        fma(Uz, Vx, -fma(Ux, Vz, 0)),  # w_y = (Uz * Vx - Ux * Vz)
        fma(Ux, Vy, -fma(Uy, Vx, 0))   # w_z = (Ux * Vy - Uy * Vx)
    ])


def test_cross():
    # Cross product of two perpendicular vectors
    v1 = Vector([0, 0, 1])
    v2 = Vector([1, 0, 0])
    assert cross(v1, v2).values == [0, 1, 0]

    # Cross product of two parallel vectors
    v1 = Vector([1, 0, 0])
    v2 = Vector([2, 0, 0])
    assert cross(v1, v2).values == [0, 0, 0]

    # Cross product of two arbitrary vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert cross(v1, v2).values == [-3, 6, -3]

    # Cross product of two arbitrary vectors
    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert cross(v1, v2).values == [-3, 6, -3]

    # Cross product of two arbitrary vectors
    v1 = Vector([4, 2, -3])
    v2 = Vector([-2, -5, 16])
    assert cross(v1, v2).values == [17, -58, -16]


def main():
    try:
        test_cross()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
