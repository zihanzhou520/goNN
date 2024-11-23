package nn

import (
	"encoding/json"
	"fmt"
	"os"
)

func (m *Model) SaveModel(filepath string) error {
	data := map[string][]Layer{
		"Layers": m.Layers,
	}

	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(data)
}

func LoadModel(filepath string) (*Model, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var rawModel struct {
		Layers []json.RawMessage
	}
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&rawModel); err != nil {
		return nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	model := &Model{}

	for _, rawLayer := range rawModel.Layers {
		var layerType struct {
			Type string
		}
		if err := json.Unmarshal(rawLayer, &layerType); err != nil {
			return nil, fmt.Errorf("failed to read layer type: %w", err)
		}

		var layer Layer
		switch layerType.Type {
		case "dense":
			var denseLayer DenseLayer
			if err := json.Unmarshal(rawLayer, &denseLayer); err != nil {
				return nil, fmt.Errorf("failed to unmarshal dense layer: %w", err)
			}
			layer = &denseLayer
		default:
			return nil, fmt.Errorf("unknown layer type: %s", layerType.Type)
		}

		model.Layers = append(model.Layers, layer)
	}

	return model, nil
}
