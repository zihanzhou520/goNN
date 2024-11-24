package dataloader

import (
	m "github.com/RSYashwanth/goNN/pkg/utils/matrix"
	// "fmt"
)

type Data struct {
	Input  *m.Matrix
	Target *m.Matrix
}

type DataLoader struct {
	Data         []Data
	CurrentBatch int
	BatchSize    int
}

func NewDataLoader(data []Data, batchSize int) *DataLoader {
	return &DataLoader{
		Data:         data,
		CurrentBatch: 0,
		BatchSize:    batchSize,
	}
}

func (d *DataLoader) GetBatch() []Data {
	start := d.CurrentBatch * d.BatchSize
	end := start + d.BatchSize
	if end > len(d.Data) {
		end = len(d.Data)
	}
	// Go back to the start
	d.CurrentBatch++
	if d.CurrentBatch >= len(d.Data)/d.BatchSize {
		d.CurrentBatch = 0
	}
	// fmt.Println(d.Data[start:end])
	// for i := range d.Data[start:end] {
	// 	fmt.Println(d.Data[start:end][i].Input)
	// 	fmt.Println(d.Data[start:end][i].Target)
	// }
	return d.Data[start:end]
}
