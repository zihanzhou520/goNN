package utils

import "fmt"

type Tensor struct {
	Data  []float64
	Shape []int
}

// Returns a new tensor of the given shape
// and with the given data in row-major order
func NewTensor(data []float64, shape []int) (*Tensor, error) {
	expectedSize := 1
	for _, i := range shape {
		expectedSize *= i
	}
	if expectedSize != len(data) {
		return nil, fmt.Errorf(
			"data size (%d) did not match shape size (%d)",
			len(data),
			expectedSize,
		)
	}
	return &Tensor{Data: data, Shape: shape}, nil
}

// Reshape reshapes the tensor into a new shape, but keeps the same data.
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	expectedSize := 1
	for _, dim := range newShape {
		expectedSize *= dim
	}

	if expectedSize != len(t.Data) {
		return nil, fmt.Errorf("new shape is incompatible with the number of elements")
	}

	t.Shape = newShape
	return &Tensor{Data: t.Data, Shape: newShape}, nil
}

// Scalar Multiplys the tensor by a scalar
func (t *Tensor) ScalarMultiply(scalar float64) *Tensor {
	newData := make([]float64, len(t.Data))
	for i := range t.Data {
		newData[i] = t.Data[i] * scalar
	}
	return &Tensor{Data: newData, Shape: t.Shape}
}

// Add tensor
func (t *Tensor) Add(t2 *Tensor) (*Tensor, error) {

	if len(t.Shape) != len(t2.Shape) {
		return nil, fmt.Errorf("cannot add tensors of different shapes")
	}

	for i := range t.Shape {
		if t.Shape[i] != t2.Shape[i] {
			return nil, fmt.Errorf("cannot add tensors of different shapes")
		}
	}

	if len(t.Data) != len(t2.Data) {
		return nil, fmt.Errorf("cannot add tensors of different sizes")
	}

	newData := make([]float64, len(t.Data))
	for i := range t.Data {
		newData[i] = t.Data[i] + t2.Data[i]
	}
	return &Tensor{Data: newData, Shape: t.Shape}, nil
}
