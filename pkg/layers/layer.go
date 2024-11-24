package layers

import (
	o "github.com/RSYashwanth/goNN/pkg/optimizers"
	m "github.com/RSYashwanth/goNN/pkg/utils/matrix"
)

// A base layer interface. The Forward
// method takes in the input and returns a matrix output allocated
// on the heap.
// The Backward method takes in the gradient and the optimizer
// and performs inplace updates and returns the updated gradient
// which will be an inplace update for the passed in arguments.
type Layer interface {
	Forward(input *m.Matrix) (*m.Matrix, error)
	Backward(grad *m.Matrix, optimizer *o.Optimizer) (*m.Matrix, error)
	Type() string
}
