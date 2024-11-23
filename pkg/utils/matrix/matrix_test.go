package utils

import (
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

	// Test random generator, if it works
	m2 := *NewRandomMatrix(3, 3, 1, 10)

	// TODO: Also count max frequency
	for i := 0; i < 3; i++ {
		for j := 0; i < 3; i++ {
			if m2[i][j] < 1 || m2[i][j] > 10 {
				t.Errorf("Expected 1, got %f", m2[i][j])
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

func TestMultiply(t *testing.T) {
	m1 := *NewInitMatrix(3, 2, 1)
	m2 := *NewInitMatrix(2, 3, 10)

	m3Ptr, e := MatrixMultiply(&m1, &m2)
	if e != nil {
		t.Error(e)
	}
	expectedMatrixPtr := NewInitMatrix(3, 3, 20)
	if !MatrixEquals(m3Ptr, expectedMatrixPtr) {
		t.Errorf("Expected %v, got %v", *expectedMatrixPtr, *m3Ptr)
	}
}
