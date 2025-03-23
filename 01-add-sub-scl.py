from lib import Matrix, Vector


def test_scl():
    v1 = Vector([1, 2, 3])
    v1.scl(2)
    assert v1.values == [2, 4, 6]

    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m1.scl(2)
    assert m1.values == [[2.0, 4.0], [6.0, 8.0]]


def test_sub():
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 3])
    v1.sub(v2)
    assert v1.values == [0, 0, 0]

    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 8.0], [6.0, 8.0]])
    m1.sub(m2)
    assert m1.values == [[-4.0, -6.0], [-3.0, -4.0]]


def test_add():
    v1 = Vector([1, 2, 3])
    v2 = Vector([1, 2, 3])
    v1.add(v2)
    assert v1.values == [2, 4, 6]

    m1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
    m2 = Matrix([[5.0, 8.0], [6.0, 8.0]])
    m1.add(m2)
    assert m1.values == [[6.0, 10.0], [9.0, 12.0]]


def main():
    test_add()
    print("All tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError:
        print("Some tests failed.")
