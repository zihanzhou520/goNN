package util

import (
	"fmt"
	"math/rand"
)

// Matrix is a type to store a float64 matrix
type Matrix [][]float64

// NewMatrix is a function to create a new matrix and init with 0
func NewMatrix(rows, cols int) Matrix {
	matrix := make(Matrix, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

// RandomMatrix is a function to create a new matrix and init with random values
func RandomMatrix(rows, cols int, min, max float64) Matrix {
	matrix := NewMatrix(rows, cols)
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = min + rand.Float64()*(max-min)
		}
	}
	return matrix
}

// Initialize is a function to set each element in a matrix to a given value
func Initialize(m Matrix, value float64) {
	for i := range m {
		for j := range m[i] {
			m[i][j] = value
		}
	}
}

// Add is a function to either broadcast a row into an entire matrix or add 2 matrices element wise
func Add(a, b Matrix) (Matrix, error) {
	rowsA, colsA := len(a), len(a[0])
	rowsB, colsB := len(b), len(b[0])

	if rowsA != rowsB && rowsB != 1 {
		return nil, fmt.Errorf("row mismatch: cannot add %dx%d to %dx%d", rowsA, colsA, rowsB, colsB)
	}
	if colsA != colsB {
		return nil, fmt.Errorf("column mismatch: cannot add %dx%d to %dx%d", rowsA, colsA, rowsB, colsB)
	}

	result := NewMatrix(rowsA, colsA)
	for i := 0; i < rowsA; i++ {
		for j := 0; j < colsA; j++ {
			if rowsB == 1 {
				result[i][j] = a[i][j] + b[0][j]
			} else {
				result[i][j] = a[i][j] + b[i][j]
			}
		}
	}
	return result, nil
}

// Subtract is a function to subtract values of 2 matrices element wise
func Subtract(a, b Matrix) (Matrix, error) {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return nil, fmt.Errorf("matrix dimensions must match for subtraction")
	}

	result := NewMatrix(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] - b[i][j]
		}
	}
	return result, nil
}

// Multiply is a function to perform matrix multiplication between 2 matrices
func Multiply(a, b Matrix) (Matrix, error) {
	if len(a[0]) != len(b) {
		return nil, fmt.Errorf("number of columns in A must equal number of rows in B")
	}

	result := NewMatrix(len(a), len(b[0]))
	for i := range result {
		for j := range result[i] {
			for k := range a[0] {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return result, nil
}

// MultiplyScalar is a function to multiple a scalar value with each element of a matrix
func MultiplyScalar(m Matrix, scalar float64) Matrix {
	result := NewMatrix(len(m), len(m[0]))
	for i := range result {
		for j := range result[i] {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

// DivideScalar is a function to divide a scalar value from each element of a matrix
func DivideScalar(m Matrix, scalar float64) Matrix {
	result := NewMatrix(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			result[i][j] = m[i][j] / scalar
		}
	}
	return result
}

// Transpose is a function to transpose a given matrix
func Transpose(m Matrix) Matrix {
	result := NewMatrix(len(m[0]), len(m))
	for i := range m {
		for j := range m[i] {
			result[j][i] = m[i][j]
		}
	}
	return result
}

// Hadamard is a function to multiply 2 matrices element wise
func Hadamard(a, b Matrix) (Matrix, error) {
	if len(a) != len(b) || len(a[0]) != len(b[0]) {
		return nil, fmt.Errorf("matrix dimensions must match for Hadamard product")
	}

	result := NewMatrix(len(a), len(a[0]))
	for i := range a {
		for j := range a[i] {
			result[i][j] = a[i][j] * b[i][j]
		}
	}
	return result, nil
}

// Sum is a function to calculate the sum of a matrix across a particular axis (0 or 1)
func Sum(m Matrix, axis int) Matrix {
	if axis == 0 {
		result := NewMatrix(1, len(m[0]))
		for _, row := range m {
			for j, val := range row {
				result[0][j] += val
			}
		}
		return result
	} else if axis == 1 {
		result := NewMatrix(len(m), 1)
		for i, row := range m {
			for _, val := range row {
				result[i][0] += val
			}
		}
		return result
	}
	panic("invalid axis: must be 0 (columns) or 1 (rows)")
}

// Row is a function to get the ith row in a matrix
func Row(m Matrix, rowIndex int) []float64 {
	if rowIndex < 0 || rowIndex >= len(m) {
		return nil
	}
	return m[rowIndex]
}

// Column is a function to get the ith column in a matrix
func Column(m Matrix, colIndex int) []float64 {
	if colIndex < 0 || colIndex >= len(m[0]) {
		return nil
	}

	col := make([]float64, len(m))
	for i := range m {
		col[i] = m[i][colIndex]
	}
	return col
}

// Apply is a function to apply a passed in function to each element in the matrix
func Apply(m Matrix, fn func(float64) float64) Matrix {
	result := NewMatrix(len(m), len(m[0]))
	for i := range m {
		for j := range m[i] {
			result[i][j] = fn(m[i][j])
		}
	}
	return result
}
