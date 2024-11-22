package nn

import (
	"fmt"
	"yashwanthrs.com/m/util"
)

type Model struct {
	Layers []Layer
}

func NewModel(layers ...Layer) *Model {
	return &Model{Layers: layers}
}

func (m *Model) Forward(input util.Matrix) (util.Matrix, error) {
	var output util.Matrix = input

	for _, layer := range m.Layers {
		output, _ = layer.Forward(output)
	}

	return output, nil
}

func (m *Model) Backward(dOutput util.Matrix) error {
	gradient := dOutput
	for i := len(m.Layers) - 1; i >= 0; i-- {
		var err error
		gradient, err = m.Layers[i].Backward(gradient)
		if err != nil {
			fmt.Println("Error in backwards propagaton: ", err)
			return err
		}
	}
	return nil
}

func (m *Model) UpdateWeights(learningRate float64) {
	for _, layer := range m.Layers {
		layer.UpdateWeights(learningRate)
	}
}

func (m *Model) Predict(input util.Matrix) (util.Matrix, error) {
	return m.Forward(input)
}
