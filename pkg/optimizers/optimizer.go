package optimizers

import m "github.com/RSYashwanth/goNN/pkg/utils/matrix"

// Out optimizer performs an inplace updation
// of the passed in parameters based on the calculated gradient and
// the respective optimizer spec
type Optimizer interface {
	Step(params *m.Matrix, grad *m.Matrix) error
}
