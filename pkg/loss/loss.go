package loss

import (
	m "demo/pkg/utils/matrix"
)

type Loss interface {
	Calculate(output, target *m.Matrix) (float64, error)
	Gradient(output, target *m.Matrix) (*m.Matrix, error)
}
