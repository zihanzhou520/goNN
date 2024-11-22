package nn

import (
	"encoding/json"
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
	return encoder.Encode(data)
}
