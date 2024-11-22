package main

import (
	"fmt"
	"log"
	"yashwanthrs.com/m/datasets"
	"yashwanthrs.com/m/nn"
	"yashwanthrs.com/m/util"
)

func main() {
	mainPath := "../../data/"
	imagePath := mainPath + "train-images.idx3-ubyte"
	labelPath := mainPath + "train-labels.idx1-ubyte"
	testImagePath := mainPath + "t10k-images.idx3-ubyte"
	testLabelPath := mainPath + "t10k-labels.idx1-ubyte"

	fmt.Println("Loading MNIST training data...")
	trainData, err := datasets.LoadMNIST(imagePath, labelPath, 6000)
	if err != nil {
		log.Fatalf("Failed to load MNIST training data: %v", err)
	}

	fmt.Println("Loading MNIST test data...")
	testData, err := datasets.LoadMNIST(testImagePath, testLabelPath, 1000)
	if err != nil {
		log.Fatalf("Failed to load MNIST test data: %v", err)
	}

	fmt.Println("Building the model...")
	model := nn.NewModel(
		nn.NewDenseLayer(784, 128, util.ReLU, util.ReLUDerivative),
		nn.NewDenseLayer(128, 64, util.ReLU, util.ReLUDerivative),
		nn.NewDenseLayer(64, 10, util.Sigmoid, util.SigmoidDerivative),
	)

	fmt.Println("Training the model...")
	err = nn.Train(
		model,
		trainData.Images, trainData.Labels,
		10,
		0.01,
		32,
	)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	fmt.Println("Evaluating the model...")
	accuracy := evaluate(model, testData.Images, testData.Labels)
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)
}

func evaluate(model *nn.Model, images [][]float64, labels [][]float64) float64 {
	correct := 0
	total := len(images)

	for i := 0; i < total; i++ {
		output, err := model.Predict([][]float64{images[i]})
		if err != nil {
			log.Printf("Prediction failed for sample %d: %v", i, err)
			continue
		}

		predicted := argMax(output[0])
		actual := argMax(labels[i])

		if predicted == actual {
			correct++
		}
	}

	return float64(correct) / float64(total)
}

func argMax(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]

	for i, value := range slice {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}

	return maxIndex
}
