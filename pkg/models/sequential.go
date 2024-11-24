package models

import (
	d "demo/pkg/dataloader"
	l "demo/pkg/layers"
	e "demo/pkg/loss"
	o "demo/pkg/optimizers"
	m "demo/pkg/utils/matrix"
	"fmt"
	"os"
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
			fmt.Fprintf(os.Stderr, "Model.Forward: Error during forward")
			return nil, err
		}
	}
	return currentInput, nil
}

func (s *Sequential) Train(dataloader *d.DataLoader,
	optimizer o.Optimizer, loss e.Loss, epochs int) error {
	var err error

	for i := 0; i < epochs; i++ {
		currentBatch := dataloader.GetBatch()
		for j, data := range currentBatch {
			currentInput := data.Input
			currentTarget := data.Target
			fmt.Printf("Epoch: %d, Batch: %d\n", i, j)
			fmt.Println(currentInput)
			fmt.Println(currentTarget)

			// Keep forwarding the input
			for _, layer := range s.Layers {
				currentInput, err = layer.Forward(currentInput)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Model.Train: Error during forward")
					return err
				}
			}

			fmt.Println(currentInput)
			// Now we will calculate the loss
			lossValue, err := loss.Calculate(currentInput, currentTarget)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Model.Train: Error during loss calculation")
				return err
			}
			fmt.Println("Loss: ", lossValue)
			// Now we will backpropagate
			currentGrad, err := loss.Gradient(currentInput, currentTarget)
			fmt.Println(currentGrad)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Model.Train: Error during loss gradient calculation")
				return err
			}
			for i := len(s.Layers) - 1; i >= 0; i-- {
				currentGrad, err = s.Layers[i].Backward(currentGrad, &optimizer)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Model.Train: Error during backpropogation")
					return err
				}
			}

		}

	}
	return nil
}
