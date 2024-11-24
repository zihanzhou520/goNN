package optimizers

import (
	m "demo/pkg/utils/matrix"
	"fmt"
)

// Simple SGD optimizer which takes in a learning LearningRate
// to perform gradient descent
type SGD struct {
	LearningRate float64
}

// Performs an inplace updation for
// the passed in parameters based on the calculated gradient
func (s *SGD) Step(params *m.Matrix, grad *m.Matrix) error {

	// Check sizes and return error if they not same

	if len(*params) != len(*grad) {
		return fmt.Errorf("params and grad must be the same size")
	}

	for i := range *params {
		if len((*params)[i]) != len((*grad)[i]) {
			return fmt.Errorf("params and grad must be the same size")
		}
		for j := range (*params)[i] {
			(*params)[i][j] -= s.LearningRate * (*grad)[i][j]
		}
	}
	return nil
}
