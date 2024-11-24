package layers

import (
	"fmt"
	o "github.com/RSYashwanth/goNN/pkg/optimizers"
	m "github.com/RSYashwanth/goNN/pkg/utils/matrix"
)

type ReluLayer struct {
	input *m.Matrix
}

func NewReluLayer() *ReluLayer {
	return &ReluLayer{
		input: nil,
	}
}

func (l *ReluLayer) Forward(input *m.Matrix) (*m.Matrix, error) {
	l.input = input
	output := make(m.Matrix, len(*input))
	for i := range *input {
		output[i] = make([]float64, len((*input)[i]))
		for j := range (*input)[i] {
			if (*input)[i][j] < 0 {
				output[i][j] = 0
			} else {
				output[i][j] = (*input)[i][j]
			}
		}
	}
	return &output, nil
}

func (l *ReluLayer) Backward(grad *m.Matrix, optimizer *o.Optimizer) (*m.Matrix, error) {

	// For ReLU we don't have any trainable params
	// so we simply compute the backward gradient
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
			if (*l.input)[i][j] < 0 {
				backwardGrad[i][j] = 0
			} else {
				backwardGrad[i][j] = (*grad)[i][j] * (*l.input)[i][j]
			}
		}
	}
	return &backwardGrad, nil
}

func (l *ReluLayer) Type() string {
	return "relu"
}
