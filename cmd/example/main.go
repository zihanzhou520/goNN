package main

import (
	"github.com/RSYashwanth/goNN/utils"
)

func main() {
	m, _ := utils.NewMatrix([]float64{1, 2, 3, 4}, []int{2, 2})
	m.Print()
}
