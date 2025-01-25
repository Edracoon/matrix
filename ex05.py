"""
Cosine: [https://fr.wikipedia.org/wiki/Cosinus]

The cosine of an angle is the ratio of the length
of the adjacent side to the length of the hypotenuse.

    cos(θ) = dot(a, b) / ( ||a|| * ||b|| )

Where:
    - dot(a, b) is the dot product of the two vectors. (See ex03.py)
    - ||a|| and ||b|| are the Euclidean norms (2-norm) of the vectors.
     - θ is the angle between the two vectors a and b.

What Does This Mean?
    If cos(θ) =  1 → The vectors point in the same direction.
    If cos(θ) = -1 → The vectors point in opposite directions.
    If cos(θ) =  0 → The vectors are perpendicular (90° angle).
    If cos(θ) is between -1 and 1 → The vectors form some intermediate angle.

In simple terms, the cosine function tells how aligned two vectors,
a bit like the dot product, but without considering the length of the vectors.
"""

from lib import Vector


def angle_cos(a: Vector, b: Vector) -> float:
    """Return the cosine of the angle between two vectors."""
    return a.dot(b) / (a.norm() * b.norm())


def test_angle_cos():
    # The vectors point in the same direction
    v1 = Vector([1, 0])
    v2 = Vector([1, 0])
    assert angle_cos(v1, v2) == 1.0

    # The vectors are perpendicular (90° angle)
    v1 = Vector([1, 0])
    v2 = Vector([0, 1])
    assert angle_cos(v1, v2) == 0.0

    # The vectors point in the exact opposite directions
    v1 = Vector([-1, 1])
    v2 = Vector([1, -1])
    assert angle_cos(v1, v2) == -0.9999999999999998
    # The norm calculation uses math.sqrt(), which introduces tiny errors

    v1 = Vector([1, 2, 3])
    v2 = Vector([4, 5, 6])
    assert angle_cos(v1, v2) == 0.9746318461970762


def main():
    try:
        test_angle_cos()
        print("All tests passed.")
    except AssertionError:
        print("The test failed")


if __name__ == "__main__":
    main()
