package main

import (
	d "demo/pkg/dataloader"
	layers "demo/pkg/layers"
	loss "demo/pkg/loss"
	models "demo/pkg/models"
	o "demo/pkg/optimizers"
	m "demo/pkg/utils/matrix"
	"fmt"
	"os"
)

func main() {
	fmt.Println("===================================")
	fmt.Println("      Neural Network Example       ")
	fmt.Println("===================================\n")

	fmt.Println("Problem: Solving the XOR Problem\n")

	fmt.Println("Network Structure:")
	fmt.Println(" - 2 by 3 Dense Layer")
	fmt.Println(" - ReLU Activation Layer")
	fmt.Println(" - 3 by 1 Dense Layer\n")

	fmt.Println("Expected Training Labels:")
	fmt.Println("  Input |  Output")
	fmt.Println("------------------")
	fmt.Println("  0, 0  |  0")
	fmt.Println("  0, 1  |  1")
	fmt.Println("  1, 0  |  1")
	fmt.Println("  1, 1  |  0")

	data := []d.Data{
		{Input: m.NewGivenMatrix([][]float64{{0}, {0}}), Target: m.NewGivenMatrix([][]float64{{0}})},
		{Input: m.NewGivenMatrix([][]float64{{0}, {1}}), Target: m.NewGivenMatrix([][]float64{{1}})},
		{Input: m.NewGivenMatrix([][]float64{{1}, {0}}), Target: m.NewGivenMatrix([][]float64{{1}})},
		{Input: m.NewGivenMatrix([][]float64{{1}, {1}}), Target: m.NewGivenMatrix([][]float64{{0}})},
	}
	dataLoader := d.NewDataLoader(data, 1)

	// Create the model
	model := models.NewSequential(
		layers.NewDenseLayer(2, 3),
		layers.NewReluLayer(),
		layers.NewDenseLayer(3, 1),
		layers.NewReluLayer(),
	)

	optim := o.NewSGD(0.1)
	loss := loss.NewMSELoss()

	// Train the model
	fmt.Println("\nTraining...")
	err := model.Train(dataLoader, optim, loss, 10)

	if err != nil {
		fmt.Fprintf(os.Stderr, "Model.Train: Error during training%v\n", err)
		return
	}
	output, err := model.Forward(m.NewGivenMatrix([][]float64{{0}, {1}}))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Model.Forward: Error during forward")
		return
	}

	fmt.Println("\nInput:")
	fmt.Println(m.NewGivenMatrix([][]float64{{1}, {1}}))
	fmt.Println("\nOutput:")
	fmt.Println(output)
}
