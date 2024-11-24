package dataloader

import (
	m "demo/pkg/utils/matrix"
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
	d.CurrentBatch++
	return d.Data[start:end]
}
