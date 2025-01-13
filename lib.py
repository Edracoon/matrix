from typing import List, Tuple, TypeVar, Generic


T = TypeVar('T')


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

    def to_matrix(self, rows: int, cols: int) -> 'Matrix':
        """Reshape a vector into a matrix with the given rows and columns."""
        if self.size() != rows * cols:
            raise ValueError("Invalid dimensions for reshaping.")

        reshaped_values = [
            self.values[i * cols:(i + 1) * cols]
            for i in range(rows)
        ]
        return Matrix(reshaped_values)

    def add(self, other: 'Vector') -> 'Vector':
        """Addition of two vectors element-wise."""
        if self.size() != other.size():
            raise ValueError("Vectors must have the same size.")
        self.values = [
            self.values[i] + other.values[i]
            for i in range(self.size())
        ]
        return self

    def sub(self, other: 'Vector') -> 'Vector':
        """Subtraction of a vector by another vector"""
        if self.size() != other.size():
            raise ValueError("Vectors must have the same size.")

        self.values = [
            self.values[i] - other.values[i]
            for i in range(self.size())
        ]
        return self

    def scl(self, scalar: T) -> 'Vector':
        """Scaling of a vector by a scalar (multiplication)"""
        self.values = [self.values[i] * scalar for i in range(self.size())]
        return self


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

    def to_vector(self) -> 'Vector':
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

    def add(self, other: 'Matrix') -> 'Matrix':
        """Add two matrices element-wise."""
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape.")

        for x in range(len(self.values)):
            for y in range(len(self.values[x])):
                self.values[y][x] += other.values[y][x]
        return self

    def sub(self, other: 'Matrix') -> 'Matrix':
        """Substration of a matrix by another matrix"""
        if self.shape() != other.shape():
            raise ValueError("Matrices must have the same shape.")

        for y in range(len(self.values)):
            for x in range(len(self.values[y])):
                self.values[y][x] -= other.values[y][x]
        return self

    def scl(self, scalar: T) -> 'Matrix':
        """Scaling of matrix by a scalar (multiplication)"""
        for y in range(len(self.values)):
            for x in range(len(self.values[y])):
                self.values[y][x] *= scalar
        return self
