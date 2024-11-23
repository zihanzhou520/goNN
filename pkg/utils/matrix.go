package utils

import "fmt"

// Can optimize later?
type Matrix [][]float64

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
	return &Matrix{Data: data, Shape: shape}, nil
}

// ScalarMultiply multiplies the matrix by a scalar.
func (m *Matrix) ScalarMultiply(scalar float64) *Matrix {
	newData := make([]float64, len(m.Data))
	for i := range m.Data {
		newData[i] = m.Data[i] * scalar
	}
	return &Matrix{Data: newData, Shape: m.Shape}
}

// Add adds two matrices together element-wise.
func (m *Matrix) Add(m2 *Matrix) (*Matrix, error) {
	if m.Shape[0] != m2.Shape[0] || m.Shape[1] != m2.Shape[1] {
		return nil, fmt.Errorf("cannot add matrices of different shapes")
	}

	newData := make([]float64, len(m.Data))
	for i := range m.Data {
		newData[i] = m.Data[i] + m2.Data[i]
	}

	return &Matrix{Data: newData, Shape: m.Shape}, nil
}

// Add adds two matrices together element-wise.
func (m *Matrix) Subtract(m2 *Matrix) (*Matrix, error) {
	if m.Shape[0] != m2.Shape[0] || m.Shape[1] != m2.Shape[1] {
		return nil, fmt.Errorf("cannot add matrices of different shapes")
	}

	newData := make([]float64, len(m.Data))
	for i := range m.Data {
		newData[i] = m.Data[i] - m2.Data[i]
	}

	return &Matrix{Data: newData, Shape: m.Shape}, nil
}

// Multiply multiplies two matrices together element-wise.
func (m *Matrix) Multiply(m2 *Matrix) (*Matrix, error) {
	if m.Shape[1] != m2.Shape[0] {
		return nil, fmt.Errorf("cannot multiply matrices of incompatible shapes")
	}
	newData := make([]float64, m.Shape[0]*m2.Shape[1])

	// Multiply using divide and conquer
	return &Matrix{Data: newData, Shape: []int{m.Shape[0], m2.Shape[1]}}, nil
}

// Print prints the matrix in row-major order.
func (m *Matrix) Print() {
	for i := 0; i < m.Shape[0]; i++ {
		for j := 0; j < m.Shape[1]; j++ {
			fmt.Printf("%f ", m.Data[i*m.Shape[1]+j])
		}
		fmt.Println()
	}
}
