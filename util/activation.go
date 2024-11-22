package util

import (
	"math"
)

// ReLU is an activation function that returns a value passed in if it is positive and returns 0 otherwise
func ReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// ReLUDerivative is the derivative of the ReLU function
func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// Sigmoid is an activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// SigmoidDerivative is the derivative of the Sigmoid activation
func SigmoidDerivative(x float64) float64 {
	sig := Sigmoid(x)
	return sig * (1 - sig)
}

// Softmax is a function that can be used in the output layer to calculate probabilities out of the model's predictions
func Softmax(inputs []float64) []float64 {
	expSum := 0.0
	exps := make([]float64, len(inputs))
	for i, input := range inputs {
		exps[i] = math.Exp(input)
		expSum += exps[i]
	}

	for i := range exps {
		exps[i] /= expSum
	}
	return exps
}

// SoftmaxDerivative is the derivative of the Softmax function
func SoftmaxDerivative(inputs []float64) [][]float64 {
	n := len(inputs)
	softmax := Softmax(inputs)
	derivative := make([][]float64, n)

	for i := 0; i < n; i++ {
		derivative[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				derivative[i][j] = softmax[i] * (1 - softmax[i])
			} else {
				derivative[i][j] = -softmax[i] * softmax[j]
			}
		}
	}
	return derivative
}

// ApplyActivation is a function to apply an activation function to a given matrix
func ApplyActivation(m Matrix, activationFunc func(float64) float64) Matrix {
	return Apply(m, activationFunc)
}
