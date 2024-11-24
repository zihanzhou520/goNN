package layers

import (
	o "demo/pkg/optimizers/SGD"
	m "demo/pkg/utils/matrix"
)

type Layer interface {
	Forward(input *m.Matrix) (*m.Matrix, error)
	// Param ??
	Backward(grad *m.Matrix, optimizer *o.Optimizer) (*m.Matrix, error)
	Type() string
}
