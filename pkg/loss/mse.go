package loss

import (
	m "demo/pkg/utils/matrix"
)

// Mean Squared Error Loss
type MSELoss struct {
}

func NewMSELoss() *MSELoss {
	return &MSELoss{}
}

func (l *MSELoss) Calculate(output, target *m.Matrix) (float64, error) {
	dist, err := m.L2DistanceSquared(output, target)
	if err != nil {
		return 0, err
	}
	return dist / float64(len(*output)), nil
}

func (l *MSELoss) Gradient(output, target *m.Matrix) (*m.Matrix, error) {
	// Gradient
	diff, err := m.Subtract(output, target)
	if err != nil {
		return nil, err
	}
	diff = m.ScalarMultiply(diff, 2.0/float64(len(*output)))
	return diff, nil
}
