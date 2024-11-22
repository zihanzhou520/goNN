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

func (layer *DenseLayer) Backward(dOutput util.Matrix) util.Matrix {
	dZ := util.ApplyActivation(layer.Output, layer.ActivationGrad)
	dZ, _ = util.Hadamard(dOutput, dZ)

	layer.WeightGrad, _ = util.Multiply(util.Transpose(layer.Input), dZ)
	layer.BiasGrad = util.Sum(dZ, 0)
	dInput, _ := util.Multiply(dZ, util.Transpose(layer.Weights))
	return dInput
}

func (layer *DenseLayer) UpdateWeights(learningRate float64) {
	layer.Weights, _ = util.Subtract(layer.Weights, util.MultiplyScalar(layer.WeightGrad, learningRate))
	layer.Biases, _ = util.Subtract(layer.Biases, util.MultiplyScalar(layer.BiasGrad, learningRate))
}
