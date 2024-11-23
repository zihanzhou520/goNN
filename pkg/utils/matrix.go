package utils

import (
	"fmt"
	"math/rand"
)

// Can optimize later?
type Matrix [][]float64

// NewMatrix is a function to create a new matrix and initialize it with 0
func ZeroMatrix(shape []int) (*Matrix, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected a 2D shape (rows x columns), got %v", shape)
	}
	matrix := make(Matrix, shape[0])
	for i := range matrix {
		matrix[i] = make([]float64, shape[1])
	}
	return &matrix, nil
}

// RandomMatrix is a function to create a new matrix and initialize it with random values
func RandomMatrix(shape []int, min, max float64) (*Matrix, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected a 2D shape (rows x columns), got %v", shape)
	}
	mp, _ := ZeroMatrix(shape)
	matrix := *mp

	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = min + rand.Float64()*(max-min)
		}
	}
	return &matrix, nil
}

// NewMatrix creates a new matrix from a slice of data and a shape.
func NewMatrix(data []float64, shape []int) (*Matrix, error) {
	if len(shape) != 2 {
		return nil, fmt.Errorf("expected a 2D shape (rows x columns), got %v", shape)
	}
	expectedSize := shape[0] * shape[1]
	if expectedSize != len(data) {
		return nil, fmt.Errorf(
			"data size (%d) did not match shape size (%d)",
			len(data),
			expectedSize,
		)
	}

	matrixData := make(Matrix, shape[0])
	for i := 0; i < shape[0]; i++ {
		matrixData[i] = make([]float64, shape[1])
		for j := 0; j < shape[1]; j++ {
			matrixData[i][j] = data[i*shape[1]+j]
		}
	}
	return &matrixData, nil
}

// ScalarMultiply multiplies the matrix by a scalar.
func (m *Matrix) ScalarMultiply(scalar float64) *Matrix {
	matrix := *m
	toReturn := make(Matrix, len(matrix))
	for i := 0; i < len(matrix); i++ {
		toReturn[i] = make([]float64, len(matrix[0]))
		for j := 0; j < len(matrix[0]); j++ {
			toReturn[i][j] = scalar * matrix[i][j]
		}
	}
	return &toReturn
}

// Add adds two matrices together element-wise.
func (m *Matrix) Add(m2 *Matrix) (*Matrix, error) {
	matrix1 := *m
	matrix2 := *m2
	if len(matrix1) != len(matrix2) || len(matrix1[0]) != len(matrix2[0]) {
		return nil, fmt.Errorf("cannot add matrices of different shapes")
	}

	toReturn := make(Matrix, len(matrix1))
	for i := range matrix1 {
		toReturn[i] = make([]float64, len(matrix1[0]))
		for j := range matrix1[0] {
			toReturn[i][j] = matrix1[i][j] + matrix2[i][j]
		}
	}

	return &toReturn, nil
}

// Subtract subtracts argument matrix from given matrix element-wise
func (m *Matrix) Subtract(m2 *Matrix) (*Matrix, error) {
	matrix1 := *m
	matrix2 := *m2
	if len(matrix1) != len(matrix2) || len(matrix1[0]) != len(matrix2[0]) {
		return nil, fmt.Errorf("cannot subtract matrices of different shapes")
	}

	toReturn := make(Matrix, len(matrix1))
	for i := range matrix1 {
		toReturn[i] = make([]float64, len(matrix1[0]))
		for j := range matrix1[0] {
			toReturn[i][j] = matrix1[i][j] - matrix2[i][j]
		}
	}

	return &toReturn, nil
}

// Hadamard multiplies matrices element-wise
func (m *Matrix) Hadamard(m2 *Matrix) (*Matrix, error) {
	matrix1 := *m
	matrix2 := *m2
	if len(matrix1) != len(matrix2) || len(matrix1[0]) != len(matrix2[0]) {
		return nil, fmt.Errorf("cannot calc hadamard product of matrices of different shapes")
	}

	toReturn := make(Matrix, len(matrix1))
	for i := range matrix1 {
		toReturn[i] = make([]float64, len(matrix1[0]))
		for j := range matrix1[0] {
			toReturn[i][j] = matrix1[i][j] * matrix2[i][j]
		}
	}

	return &toReturn, nil
}

// Multiply multiplies two matrices together element-wise.
func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
	matrix1 := *m
	matrix2 := *m2

	if len(matrix1[0]) != len(matrix2) {
		return nil, fmt.Errorf("number of columns in A must equal number of rows in B")
	}

	toReturn := make(Matrix, len(matrix1))
	c := make(chan tmpRowRes, len(matrix1))

	for i := 0; i < len(matrix1); i++ {
		go calcRow(c, matrix1[i], matrix2, i)
	}

	for x := 0; x < len(matrix1); x++ {
		rowResult := <-c
		toReturn[rowResult.idx] = rowResult.result
	}

	return &toReturn, nil
}

func calcRow(c chan tmpRowRes, rowA []float64, matrixB Matrix, position int) {
	colsB := len(matrixB[0])
	resultRow := make([]float64, colsB)

	for j := 0; j < colsB; j++ {
		for k := 0; k < len(rowA); k++ {
			resultRow[j] += rowA[k] * matrixB[k][j]
		}
	}

	c <- tmpRowRes{result: resultRow, idx: position}
}

type tmpRowRes struct {
	result []float64
	idx    int
}

// Print prints the matrix in row-major order.
func (m *Matrix) Print() {
	matrix := *m
	for i := 0; i < len(matrix); i++ {
		for j := 0; j < len(matrix[0]); j++ {
			fmt.Printf("%f ", matrix[i][j])
		}
		fmt.Println()
	}
}
