package layers

import (
	m "demo/pkg/utils/matrix"
)

type Layer interface {
	Forward(input *m.Matrix) (*m.Matrix, error)
	Backward(input *m.Matrix) (*m.Matrix, error)
	Type() string
}
