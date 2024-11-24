package layers

import (
	o "demo/pkg/optimizers"
	m "demo/pkg/utils/matrix"
	"fmt"
	"math"
)

type TanhLayer struct {
	input *m.Matrix
}

func NewTanhLayer() *TanhLayer {
	return &TanhLayer{
		input: nil,
	}
}

// Forward pass: apply tanh activation to the input
func (l *TanhLayer) Forward(input *m.Matrix) (*m.Matrix, error) {
	l.input = input
	output := make(m.Matrix, len(*input))
	for i := range *input {
		output[i] = make([]float64, len((*input)[i]))
		for j := range (*input)[i] {
			output[i][j] = math.Tanh((*input)[i][j])
		}
	}
	return &output, nil
}

// Backward pass: compute gradient of tanh activation
func (l *TanhLayer) Backward(grad *m.Matrix, optimizer *o.Optimizer) (*m.Matrix, error) {
	if len(*l.input) != len(*grad) {
		return nil, fmt.Errorf("input and grad must be the same size")
	}
	backwardGrad := make(m.Matrix, len(*l.input))
	for i := range *l.input {
		if len((*l.input)[i]) != len((*grad)[i]) {
			return nil, fmt.Errorf("input and grad must be the same size")
		}
		backwardGrad[i] = make([]float64, len((*l.input)[i]))
		for j := range (*l.input)[i] {
			tanhVal := math.Tanh((*l.input)[i][j])
			backwardGrad[i][j] = (*grad)[i][j] * (1 - tanhVal*tanhVal) // Derivative of tanh(x) is 1 - tanh^2(x)
		}
	}
	return &backwardGrad, nil
}

// Return layer type
func (l *TanhLayer) Type() string {
	return "tanh"
}
