package layers

import (
	o "github.com/RSYashwanth/goNN/pkg/optimizers"
	m "github.com/RSYashwanth/goNN/pkg/utils/matrix"
)

type DenseLayer struct {
	Weights *m.Matrix
	Bias    *m.Matrix
	Input   *m.Matrix
}

// Generates a new dense layer
func NewDenseLayer(inputNeurons, outputNeurons int) *DenseLayer {
	weights := m.NewRandomMatrix(outputNeurons, inputNeurons, -1, 1)
	bias := m.NewRandomMatrix(outputNeurons, 1, -1, 1)
	return &DenseLayer{
		Weights: weights,
		Bias:    bias,
	}
}

func (l *DenseLayer) Forward(input *m.Matrix) (*m.Matrix, error) {
	l.Input = input
	output, err := m.MatrixMultiply(l.Weights, input)
	if err != nil {
		return nil, err
	}
	output, err = m.Add(output, l.Bias)
	if err != nil {
		return nil, err
	}
	return output, nil
}

func (l *DenseLayer) Backward(grad *m.Matrix, optimizer *o.Optimizer) (*m.Matrix, error) {

	// For the dense layer, first we will update the weights
	weightsGrad, err := m.MatrixMultiply(grad, l.Input.Transpose())
	if err != nil {
		return nil, err
	}
	// We will update the weights based on this gradient and our optimizer
	err = (*optimizer).Step(l.Weights, weightsGrad)
	if err != nil {
		return nil, err
	}

	// Next for the bias
	err = (*optimizer).Step(l.Bias, grad)
	if err != nil {
		return nil, err
	}

	// Lastly we return the output gradient
	var outputGrad *m.Matrix
	outputGrad, err = m.MatrixMultiply(l.Weights.Transpose(), grad)
	if err != nil {
		return nil, err
	}
	return outputGrad, nil
}

func (l *DenseLayer) Type() string {
	return "dense"
}
