package loss

import (
	m "demo/pkg/utils/matrix"
)

type Loss interface {
	Calculate(input, target *m.Matrix) (*m.Matrix, error)
	Gradient(input, target *m.Matrix) (*m.Matrix, error)
}
