from typing import List, Tuple, TypeVar, Generic
from math import fma

T = TypeVar("T")


def abs(n: T) -> T:
    return -n if n < 0 else n


# ===========================================================================
# ============================== Vector =====================================
# ===========================================================================


class Vector(Generic[T]):
    """A class representing a mathematical vector."""

    def __init__(self, values: List[T]):
        self.values = values

    def __getitem__(self, index):
        return self.values[index]

    def __setitem__(self, index, value):
        self.values[index] = value

    def __str__(self) -> str:
        """Print the vector in a readable format."""
        return "Vector: " + str(self.values)

    def size(self) -> int:
        """Return the size (length) of the vector."""
        return len(self.values)

    def to_matrix(self, rows: int, cols: int) -> "Matrix":
        """Reshape a vector into a matrix with the given rows and columns."""
        if self.size() != rows * cols:
            raise AssertionError("Invalid dimensions for reshaping.")

        reshaped_values = [
            self.values[i * cols:(i + 1) * cols]
            for i in range(rows)
        ]
        return Matrix(reshaped_values)

    def add(self, other: "Vector") -> "Vector":
        """Addition of two vectors element-wise."""
        if self.size() != other.size():
            raise ValueError("Vectors must have the same size.")
        self.values = [
            self.values[i] + other.values[i]
            for i in range(self.size())
        ]
        return self

    def sub(self, other: "Vector") -> "Vector":
        """Subtraction of a vector by another vector"""
        if self.size() != other.size():
            raise AssertionError("Vectors must have the same size.")

        self.values = [
            self.values[i] - other.values[i]
            for i in range(self.size())
        ]
        return self

    def scl(self, scalar: T) -> "Vector":
        """Scaling of a vector by a scalar (multiplication)"""
        self.values = [self.values[i] * scalar for i in range(self.size())]
        return self

    def dot(self, other: "Vector") -> T:
        """Dot product of two vectors."""
        if self.size() != other.size():
            raise AssertionError("Vectors must have the same size.")
        res = 0
        for i in range(self.size()):
            res = fma(self.values[i], other.values[i], res)
        return res

    def norm_1(self) -> float:
        """
        Return the Manhattan distance of the vector.
        The sum of the absolute values of all elements.
        """
        return sum(abs(n) for n in self.values)

    def norm(self) -> float:
        """
        Return the Euclidean distance of the vector (hypotenuse).
        The square root of the sum of the squares of all elements.
        """
        res = 0
        for val in self.values:
            res = fma(val, val, res)
        return res**0.5

    def norm_inf(self) -> float:
        """
        Return the maximum absolute value of the vector.
        When you want to know the most significant component of a vector
        """
        return max(abs(n) for n in self.values)


# ===========================================================================
# ============================== Matrix =====================================
# ===========================================================================


class Matrix(Generic[T]):
    """A class representing a mathematical matrix."""

    def __init__(self, values: List[List[T]]):
        self.values = values

    def __getitem__(self, index):
        """Override __getitem__ to allow matrix[y, x] access"""
        if isinstance(index, tuple):
            y, x = index
            return self.values[y][x]
        else:
            return self.values[index]

    def __setitem__(self, index, value):
        """Override __setitem__ to allow matrix[y, x] = value"""
        if isinstance(index, tuple):
            y, x = index
            self.values[y][x] = value
        else:
            self.values[index] = value

    def __str__(self) -> str:
        """Print the matrix in a readable format."""
        rows = ["[" + ", ".join(map(str, row)) + "]" for row in self.values]
        return "Matrix:\n" + "\n".join(rows)

    def to_vector(self) -> "Vector":
        """Reshape the matrix into a vector."""
        flattened_values = [item for row in self.values for item in row]
        return Vector(flattened_values)

    def shape(self) -> Tuple[int, int]:
        """Return the shape (rows, columns) of the matrix."""
        rows = len(self.values)
        cols = len(self.values[0]) if rows > 0 else 0
        return rows, cols

    def is_square(self) -> bool:
        """Return True if the matrix is square otherwise False."""
        rows, cols = self.shape()
        return rows == cols

    def identity_matrix(self, n) -> "Matrix":
        """Creates an identity matrix of size n x n."""
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    def add(self, other: "Matrix") -> "Matrix":
        """Add two matrices element-wise."""
        if self.shape() != other.shape():
            raise AssertionError("Matrices must have the same shape.")

        for x in range(len(self.values)):
            for y in range(len(self.values[x])):
                self.values[y][x] += other.values[y][x]
        return self

    def sub(self, other: "Matrix") -> "Matrix":
        """Substration of a matrix by another matrix"""
        if self.shape() != other.shape():
            raise AssertionError("Matrices must have the same shape.")

        for y in range(len(self.values)):
            for x in range(len(self.values[y])):
                self.values[y][x] -= other.values[y][x]
        return self

    def scl(self, scalar: T) -> "Matrix":
        """Scaling of matrix by a scalar (multiplication)"""
        for y in range(len(self.values)):
            for x in range(len(self.values[y])):
                self.values[y][x] *= scalar
        return self

    def __scl_row(self, row, scalar: T) -> "Matrix":
        """Scales a row by a factor."""
        self[row] = [element * scalar for element in self[row]]

    def mul_vec(self, vec: Vector) -> Vector:
        """Multiply a vector by a matrix."""
        if vec.size() != self.shape()[0]:
            raise ValueError("Vector size must match the matrix row size.")

        result = Vector([0.0 for _ in range(vec.size())])
        for i in range(self.shape()[0]):
            for j in range(vec.size()):
                result[i] = fma(self[i, j], vec[j], result[i])
        return result

    def mul_mat(self, mat: "Matrix") -> "Matrix":
        if self.shape()[1] != mat.shape()[0]:
            raise ValueError("Dimensions are incompatible for multiplication.")
        # Init result matrix
        result = Matrix([
            [0.0 for _ in range(self.shape()[0])]
            for _ in range(mat.shape()[1])
        ])
        # Iterate self rows
        for i in range(self.shape()[0]):
            # Iterate mat columns
            for j in range(mat.shape()[1]):
                # Iterate over the shared dimension
                for k in range(mat.shape()[0]):
                    result[i, j] = fma(self[i, k], mat[k, j], result[i, j])
        return result

    def trace(self) -> "Matrix":
        if self.shape()[0] != self.shape()[1]:
            raise ValueError("Trace is only available for squared matrices")

        res = 0
        for i in range(self.shape()[0]):
            res += self[i, i]
        return res

    def transpose(self) -> "Matrix":
        return Matrix([
            [self[j, i] for j in range(self.shape()[0])]
            for i in range(self.shape()[1])
        ])

    def __find_nonzero_row(m, curr_row, col):
        rows = m.shape()[0]
        for row in range(curr_row, rows):
            if m[row, col] != 0:
                return row
        return None

    def __row_swap(m, i, j):
        """Swaps two rows in a matrix."""
        temp = m[i]
        m[i] = m[j]
        m[j] = temp

    def __make_pivot_one(m, curr_row, col):
        pivot_element = m[curr_row, col]
        for i in range(len(m[curr_row])):
            m[curr_row, i] /= pivot_element

    def __eliminate_below(m, curr_row, col):
        nrows, ncols = m.shape()
        for row in range(curr_row + 1, nrows):
            factor = m[row, col]
            for i in range(ncols):
                m[row, i] -= factor * m[curr_row, i]

    def row_echelon(self) -> "Matrix":
        matrix = Matrix(self.values)
        ncols = matrix.shape()[1]
        row = 0
        # If matrix has 3 columns this loop will run for 3 times
        for col in range(ncols):
            nonzero_row = matrix.__find_nonzero_row(row, col)
            if nonzero_row is None:
                continue
            # When finding a non zero row we operate those 3 steps
            matrix.__row_swap(row, nonzero_row)
            matrix.__make_pivot_one(row, col)
            matrix.__eliminate_below(row, col)
            row += 1
        return matrix

    def is_row_echelon_form(self):
        rows, cols = self.shape()
        previous_one = -1

        for row in range(rows):
            found_one = False
            for col in range(cols):
                if self[row, col] == 0:
                    continue
                if col <= previous_one:
                    return False
                previous_one = col
                found_one = True
                break
            if not found_one and any(
                    self[row, col] != 0 for col in range(cols)):
                return False
        return True

    def __determinant_sub_matrix(m, currCol) -> 'Matrix':
        ncols = m.shape()[0]
        sub = Matrix([[0] * (ncols - 1) for _ in range(ncols - 1)])
        for i in range(1, ncols):
            subcol = 0
            for j in range(ncols):
                # Skip the current column
                if j == currCol:
                    continue
                # Fill the submatrix
                sub[i - 1][subcol] = m[i][j]
                subcol += 1
        return sub

    def determinant(self) -> T:
        """Computes the determinant of the matrix using cofactor expansion"""
        ncols = self.shape()[0]
        # If the matrix is 1x1
        if ncols == 1:
            return self[0][0]
        # For 2x2 matrix
        # (recursive endpoint for larger matrices)
        if ncols == 2:
            return self[0][0] * self[1][1] - \
                self[0][1] * self[1][0]
        # Recursive case for larger matrices
        res = 0
        for col in range(ncols):
            # Reduce the matrix size
            # removing the first row and the current column
            sub_mat = self.__determinant_sub_matrix(col)
            # Cofactor expansion
            sign = 1 if col % 2 == 0 else -1
            res += sign * self[0][col] * sub_mat.determinant()

        return res

    def __add_scaled_row(self, target_row, source_row, factor):
        """Adds a scaled row to another row."""
        self[target_row] = [
            target + factor * source
            for target, source in zip(self[target_row], self[source_row])
        ]

    def inverse(self):
        """Computes the inverse of the matrix using Gaussian elimination."""
        n = self.shape()[0]
        # Create augmented matrix [A | ID]
        # Deep copy to prevent modification of the original matrix
        A = Matrix([row[:] for row in self.values])
        ID = self.identity_matrix(n)

        # Forward elimination
        for i in range(n):
            # Find pivot
            if A[i][i] == 0:
                for j in range(i + 1, n):
                    if A[j][i] != 0:
                        A.__row_swap(i, j)
                        ID.__row_swap(i, j)
                        break
                else:
                    raise ValueError("Matrix cannot be inverted (singular).")

            # Scale pivot row to make pivot = 1
            factor = 1 / A[i][i]
            A.__scl_row(i, factor)
            ID.__scl_row(i, factor)

            # Make zeros below pivot
            for j in range(i + 1, n):
                factor = -A[j][i]
                A.__add_scaled_row(j, i, factor)
                ID.__add_scaled_row(j, i, factor)

        # Backward elimination
        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                factor = -A[j][i]
                A.__add_scaled_row(j, i, factor)
                ID.__add_scaled_row(j, i, factor)

        return ID  # The right half of the augmented matrix is now A⁻¹

    def rank(self):
        """Computes the rank of the matrix."""
        REF = self.row_echelon()
        rank = 0
        for row in REF.values:
            # If the row is not all zeros
            # it means that the row is linearly independent
            if any(row):
                rank += 1
        return rank
