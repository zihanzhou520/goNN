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
	for i := range *input {
		for j := range (*input)[i] {
			if (*input)[i][j] < 0 {
				(*input)[i][j] = 0
			}
		}
	}
	return input, nil
}

func (l *ReluLayer) Backward(input *m.Matrix) (*m.Matrix, error) {
	for i := range *input {
		for j := range (*input)[i] {
			if (*input)[i][j] < 0 {
				(*input)[i][j] = 0
			} else {
				(*input)[i][j] = 1
			}
		}
	}
	return input, nil
}

func (l *ReluLayer) Type() string {
	return "relu"
}
