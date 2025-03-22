from lib import Matrix


A = Matrix([[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]])


REF: Matrix = A.row_echelon()
print(REF)
print(REF.is_row_echelon_form())


# Tests

u = Matrix([
    [1., 0., 0.],
    [0., 1., 0.],
    [0., 0., 1.],
])
print(u.row_echelon())
print(u.is_row_echelon_form())
# [1.0, 0.0, 0.0]
# [0.0, 1.0, 0.0]
# [0.0, 0.0, 1.0]

u = Matrix([
    [1., 2.],
    [3., 4.],
])
print(u.row_echelon())
print(u.is_row_echelon_form())
# [1.0, 0.0]
# [0.0, 1.0]

u = Matrix([
    [1., 2.],
    [2., 4.],
])
print(u.row_echelon())
print(u.is_row_echelon_form())
# [1.0, 2.0]
# [0.0, 0.0]

u = Matrix([
    [8., 5., -2., 4., 28.],
    [4., 2.5, 20., 4., -4.],
    [8., 5., 1., 4., 17.],
])
print(u.row_echelon())
# [1.0, 0.625, 0.0, 0.0, -12.1666667]
# [0.0, 0.0, 1.0, 0.0, -3.6666667]
# [0.0, 0.0, 0.0, 1.0, 29.5 ]
