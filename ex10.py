from lib import Matrix


def test_row_echelon():
    # A = Matrix([[1, 2, 3],
    #             [4, 5, 6],
    #             [7, 8, 9]])
    # REF = A.row_echelon()
    # assert REF == Matrix([[1, 2, 3],
    #                       [0, -3, -6],
    #                       [0, 0, 0]])
    # assert REF.is_row_echelon_form()

    A = Matrix([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ])
    REF = A.row_echelon()
    assert REF == Matrix([[1., 0., 0.],
                          [0., 1., 0.],
                          [0., 0., 1.]])
    assert REF.is_row_echelon_form()

    A = Matrix([
        [1., 2.],
        [2., 4.],
    ])
    REF = A.row_echelon()
    assert REF == Matrix([[1., 2.],
                          [0., 0.]])
    assert REF.is_row_echelon_form()

    # A = Matrix([
    #     [8., 5., -2., 4., 28.],
    #     [4., 2.5, 20., 4., -4.],
    #     [8., 5., 1., 4., 17.],
    # ])
    # REF = A.row_echelon()
    # assert REF == Matrix([[1., 0.625, 0.0, 0.0, -12.1666667],
    #                       [0., 0.0, 1.0, 0.0, -3.6666667],
    #                       [0., 0.0, 0.0, 1.0, 29.5 ]])
    # assert REF.is_row_echelon_form()


def main():
    try:
        test_row_echelon()
        print("All tests passed.")
    except AssertionError:
        print("Some tests failed.")


if __name__ == "__main__":
    main()
