package utils

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestConstructors(t *testing.T) {
	// Test to see if the constructor works
	m := *NewZeroMatrix(3, 3)
	if len(m) != 3 {
		t.Errorf("Expected 3 rows, got %d", len(m))
	}

	for i := 0; i < 3; i++ {
		for j := 0; i < 3; i++ {
			if m[i][j] != 0 {
				t.Errorf("Expected 0, got %f", m[i][j])
			}
		}
	}

	// Test if the init constructor works
	m1 := *NewInitMatrix(3, 3, 1)
	for i := 0; i < 3; i++ {
		for j := 0; i < 3; i++ {
			if m1[i][j] != 1 {
				t.Errorf("Expected 1, got %f", m1[i][j])
			}
		}
	}

}

func TestAdd(t *testing.T) {
	// Test to see if the add works
	m1 := *NewInitMatrix(3, 3, 1)
	m2 := *NewZeroMatrix(3, 3)
	m3Ptr, e := Add(&m1, &m2)
	if e != nil {
		t.Error(e)
	}
	m3 := *m3Ptr

	for i := 0; i < 3; i++ {
		for j := 0; i < 3; i++ {
			if m3[i][j] != 1 {
				t.Errorf("Expected 1, got %f", m3[i][j])
			}
		}
	}
}

// The abs function is a helper function to get the abs value
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// The sequentialMultiply function helps fuzz the matrix multiplication test
func sequentialMultiply(a, b Matrix) (*Matrix, error) {
	result := make(Matrix, len(a))
	for i := range result {
		result[i] = make([]float64, len(b[0]))
	}

	for i := 0; i < len(a); i++ {
		for j := 0; j < len(b[0]); j++ {
			for k := 0; k < len(b); k++ {
				result[i][j] += a[i][k] * b[k][j]
			}
		}
	}

	return &result, nil
}

func TestMultiplyFuzz(t *testing.T) {
	f := func(rowsA, colsA, colsB int, minVal, maxVal float64) bool {
		rowsA = abs(rowsA % 1000)
		colsA = abs(colsA % 1000)
		colsB = abs(colsB % 1000)
		if rowsA == 0 || colsA == 0 || colsB == 0 {
			return true
		}

		if minVal > maxVal {
			minVal, maxVal = maxVal, minVal
		}

		matrixA := NewRandomMatrix(rowsA, colsA, minVal, maxVal)
		matrixB := NewRandomMatrix(colsA, colsB, minVal, maxVal)

		correctResult, err1 := sequentialMultiply(*matrixA, *matrixB)
		if err1 != nil {
			t.Errorf("Sequential multiply error: %v", err1)
			return false
		}

		concurrentResult, err2 := MatrixMultiply(matrixA, matrixB)
		if err2 != nil {
			t.Errorf("Concurrent multiply error: %v", err2)
			return false
		}

		if !MatrixEquals(correctResult, concurrentResult) {
			t.Errorf("Results do not match: sequential=%v, concurrent=%v", correctResult, concurrentResult)
			return false
		}

		return true
	}

	// Run with -v for iteration status
	it := 100
	for i := 0; i < it; i++ {
		fmt.Printf("\rIteration: %d/%d", i+1, it)
		f(rand.Intn(700), rand.Intn(700), rand.Intn(700), 10.0, 10.0)
	}
}
