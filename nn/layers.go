package nn

import (
	"yashwanthrs.com/m/util"
)

type Layer interface {
	Forward(input util.Matrix) (util.Matrix, error)
	Backward(dOutput util.Matrix) (util.Matrix, error)
	UpdateWeights(learningRate float64)
}
