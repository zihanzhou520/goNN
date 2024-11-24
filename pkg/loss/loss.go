package loss

import (
	m "github.com/RSYashwanth/goNN/pkg/utils/matrix"
)

type Loss interface {
	Calculate(output, target *m.Matrix) (float64, error)
	Gradient(output, target *m.Matrix) (*m.Matrix, error)
}
