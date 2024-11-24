package layers

import (
	m "demo/pkg/utils/matrix"
)

type ReluLayer struct {
}

func NewReluLayer() *ReluLayer {
	return &ReluLayer{}
}

func (l *ReluLayer) Forward(input *m.Matrix) (*m.Matrix, error) {
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

func (l *ReluLayer) Backward(input *m.Matrix) (*m.Matrix, error) {
	output := make(m.Matrix, len(*input))
	for i := range *input {
		output[i] = make([]float64, len((*input)[i]))
		for j := range (*input)[i] {
			if (*input)[i][j] < 0 {
				output[i][j] = 0
			} else {
				output[i][j] = 1
			}
		}
	}
	return &output, nil
}

func (l *ReluLayer) Type() string {
	return "relu"
}
