package models

import (
	l "demo/pkg/layers"
	e "demo/pkg/loss"
	o "demo/pkg/optimizers"
	m "demo/pkg/utils/matrix"
)

type Sequential struct {
	Layers []l.Layer
}

// Returns a new Sequential model
func NewSequential(layers ...l.Layer) *Sequential {
	return &Sequential{
		Layers: layers,
	}
}

func (s *Sequential) Forward(input *m.Matrix) (*m.Matrix, error) {

	currentInput := input
	var err error
	for _, layer := range s.Layers {
		currentInput, err = layer.Forward(currentInput)
		if err != nil {
			return nil, err
		}
	}
	return currentInput, nil
}

func (s *Sequential) Train(input *m.Matrix, output *m.Matrix,
	optimizer *o.Optimizer, loss *e.Loss, epochs int) (*m.Matrix, error) {
	currentInput := input
	var err error
	for _, layer := range s.Layers {
		currentInput, err = layer.Forward(currentInput)
		if err != nil {
			return nil, err
		}
		currentInput, err = layer.Backward(output, optimizer)
		if err != nil {
			return nil, err
		}
	}
	return currentInput, nil
}
