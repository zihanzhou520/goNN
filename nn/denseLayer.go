package nn

import (
	"fmt"
	"yashwanthrs.com/m/util"
)

type DenseLayer struct {
	Weights        util.Matrix
	Biases         util.Matrix
	Input          util.Matrix
	Output         util.Matrix
	WeightGrad     util.Matrix
	BiasGrad       util.Matrix
	Activation     func(float64) float64
	ActivationGrad func(float64) float64
}

// NewDenseLayer is a function to create a new dense layer
func NewDenseLayer(inputSize, outputSize int, activation func(float64) float64, activationGrad func(float64) float64) *DenseLayer {
	weights := util.RandomMatrix(inputSize, outputSize, -0.1, 0.1)
	biases := util.NewMatrix(1, outputSize)

	return &DenseLayer{
		Weights:        weights,
		Biases:         biases,
		Activation:     activation,
		ActivationGrad: activationGrad,
	}
}

// Forward is a function to simulate forward propagation
func (layer *DenseLayer) Forward(input util.Matrix) (util.Matrix, error) {
	layer.Input = input

	z, err := util.Multiply(input, layer.Weights)
	if err != nil {
		fmt.Println("Error in Multiply:", err)
		return nil, err
	}

	z, err = util.Add(z, layer.Biases)
	if err != nil {
		fmt.Println("Error in Add:", err)
		return nil, err
	}

	layer.Output = util.Apply(z, layer.Activation)
	return layer.Output, nil
}

// Backward is a function to apply backward propagation to this layer
func (layer *DenseLayer) Backward(dOutput util.Matrix) (util.Matrix, error) {
	dZ := util.ApplyActivation(layer.Output, layer.ActivationGrad)
	dZ, err := util.Hadamard(dOutput, dZ)
	if err != nil {
		fmt.Println("Error in Hadamard: ", err)
		return nil, err
	}

	layer.WeightGrad, _ = util.Multiply(util.Transpose(layer.Input), dZ)
	layer.BiasGrad = util.Sum(dZ, 0)
	dInput, err := util.Multiply(dZ, util.Transpose(layer.Weights))
	if err != nil {
		fmt.Println("Error in Multiply: ", err)
		return nil, err
	}
	return dInput, nil
}

func (layer *DenseLayer) UpdateWeights(learningRate float64) error {
	var err error
	layer.Weights, err = util.Subtract(layer.Weights, util.MultiplyScalar(layer.WeightGrad, learningRate))
	if err != nil {
		fmt.Println("Error in Subtract: ", err)
		return err
	}

	layer.Biases, err = util.Subtract(layer.Biases, util.MultiplyScalar(layer.BiasGrad, learningRate))
	if err != nil {
		fmt.Println("Error in Subtract: ", err)
		return err
	}
	return nil
}
