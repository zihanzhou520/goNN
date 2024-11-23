package utils

import (
	"fmt"
	"math/rand"
)

// Can optimize later?
type Matrix [][]float64

///////////////////////////////
// Matrix Constructors
///////////////////////////////

// A simple constructor which will create a matrix of zeros
func NewZeroMatrix(rows, cols int) *Matrix {
	matrix := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
	}
	return &matrix
}

// A simple constructor which will create a matrix of given initial values
func NewInitMatrix(rows, cols int, val float64) *Matrix {
	matrix := make(Matrix, rows)
	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			matrix[i][j] = float64(val)
		}
	}
	return &matrix
}

// A simple constructor which will create a matrix of random numbers
func NewRandomMatrix(rows, cols int, minVal, maxVal float64) *Matrix {
	matrix := *NewZeroMatrix(rows, cols)
	for i := range matrix {
		for j := range matrix[i] {
			matrix[i][j] = minVal + rand.Float64()*(maxVal-minVal)
		}
	}
	return &matrix
}

///////////////////////////////
// Matrix Operations
///////////////////////////////

// Add adds two matrices together element-wise.
func Add(m1Ptr, m2Ptr *Matrix) (*Matrix, error) {
	m1 := *m1Ptr
	m2 := *m2Ptr

	a1 := len(m1)
	a2 := len(m1[0])
	b1 := len(m2)
	b2 := len(m2[0])

	if a1 != b1 || a2 != b2 {
		return nil, fmt.Errorf("cannot add matrices of different shapes")
	}

	newData := make(Matrix, a1)

	for i := 0; i < a1; i++ {
		newData[i] = make([]float64, a2)
		for j := 0; j < a2; j++ {
			newData[i][j] = m1[i][j] + m2[i][j]
		}
	}
	return &newData, nil
}

// Subtracts two matrices element-wise
func Subtract(m1Ptr, m2Ptr *Matrix) (*Matrix, error) {
	m1 := *m1Ptr
	m2 := *m2Ptr

	a1 := len(m1)
	a2 := len(m1[0])
	b1 := len(m2)
	b2 := len(m2[0])

	if a1 != b1 || a2 != b2 {
		return nil, fmt.Errorf("cannot add matrices of different shapes")
	}

	newData := make(Matrix, a1)

	for i := 0; i < a1; i++ {
		newData[i] = make([]float64, a2)
		for j := 0; j < a2; j++ {
			newData[i][j] = m1[i][j] + m2[i][j]
		}
	}
	return &newData, nil
}

// ScalarMultiply multiplies a matrix by a scalar
func ScalarMultiply(mPtr *Matrix, scalar float64) *Matrix {
	m := *mPtr
	newData := make(Matrix, len(m))
	for i := range m {
		newData[i] = make([]float64, len(m[i]))
		for j := range m[i] {
			newData[i][j] = m[i][j] * scalar
		}
	}
	return &newData
}

// MatrixMultiply multiplies two matrices
func MatrixMultiply(m1Ptr, m2Ptr *Matrix) (*Matrix, error) {
	m1 := *m1Ptr
	m2 := *m2Ptr

	a1 := len(m1)
	a2 := len(m1[0])
	b1 := len(m2)
	b2 := len(m2[0])

	if a2 != b1 {
		return nil, fmt.Errorf("cannot multiply matrices of incompatible shapes")
	}

	newData := make(Matrix, a1)
	for i := 0; i < a1; i++ {
		newData[i] = make([]float64, b2)
		for j := 0; j < b2; j++ {
			for k := 0; k < a2; k++ {
				newData[i][j] += m1[i][k] * m2[k][j]
			}
		}
	}
	return &newData, nil
}

// Transpose returns the transpose of the matrix
func Transpose(mPtr *Matrix) *Matrix {
	m := *mPtr
	newData := make(Matrix, len(m[0]))
	for i := range m[0] {
		newData[i] = make([]float64, len(m))
		for j := range m {
			newData[i][j] = m[j][i]
		}
	}
	return &newData
}

// MatrixEquals returns true if the two matrices are equal
func MatrixEquals(m1Ptr, m2Ptr *Matrix) bool {
	m1 := *m1Ptr
	m2 := *m2Ptr
	if len(m1) != len(m2) {
		return false
	}
	for i := range m1 {
		if len(m1[i]) != len(m2[i]) {
			return false
		}
		for j := range m1[i] {
			if m1[i][j] != m2[i][j] {
				return false
			}
		}
	}
	return true
}

///////////////////////////////
// Matrix Functions
///////////////////////////////

// Instances method to Add
func (m1Ptr *Matrix) Add(m2Ptr *Matrix) error {
	m1 := *m1Ptr
	m2 := *m2Ptr

	a1 := len(m1)
	a2 := len(m1[0])
	b1 := len(m2)
	b2 := len(m2[0])

	if a1 != b1 || a2 != b2 {
		return fmt.Errorf("cannot add matrices of different shapes")
	}

	for i := range m1 {
		for j := range m1[i] {
			m1[i][j] += m2[i][j]
		}
	}
	return nil
}

// Instances method to ScalarMultiply
func (mPtr *Matrix) ScalarMultiply(scalar float64) {
	m := *mPtr
	for i := range m {
		for j := range m[i] {
			m[i][j] *= scalar
		}
	}
}

// Instances method to Transpose
func (mPtr *Matrix) Transpose() *Matrix {
	m := *mPtr
	newData := make(Matrix, len(m[0]))
	for i := range m[0] {
		newData[i] = make([]float64, len(m))
		for j := range m {
			newData[i][j] = m[j][i]
		}
	}
	return &newData
}
