package utils

import (
	"testing"
)

func TestMatrix(t *testing.T) {
	_, err := NewMatrix([]float64{1, 2, 3, 4}, []int{2, 2})
	if err != nil {
		t.Error(err)
	}
}
