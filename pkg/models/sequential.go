package models

import (
	l "demo/pkg/layers"
	e "demo/pkg/loss"
	o "demo/pkg/optimizers"
	m "demo/pkg/utils/matrix"
	"fmt"
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
	optimizer *o.Optimizer, loss *e.Loss, epochs int) error {
	currentInput := input
	var err error

	for i := 0; i < epochs; i++ {
		// Keep forwarding the input
		for _, layer := range s.Layers {
			currentInput, err = layer.Forward(currentInput)
			if err != nil {
				return err
			}
		}

		fmt.Println("Loss: ", fmt.Sprintf("%.4f", (*loss).Calculate(currentInput, output)[0][0]))
		// Now we will backpropagate
		currentGrad, err := (*loss).Gradient(currentInput, output)
		if err != nil {
			return err
		}
		for i := len(s.Layers) - 1; i >= 0; i-- {
			currentGrad, err = s.Layers[i].Backward(currentGrad, optimizer)
			if err != nil {
				return err
			}
		}

	}
	return nil
}
