package main

import (
	"fmt"
	"log"
	"yashwanthrs.com/m/datasets"
	"yashwanthrs.com/m/nn"
)

func test() {
	mainPath := "../../data/"
	testImagePath := mainPath + "t10k-images.idx3-ubyte"
	testLabelPath := mainPath + "t10k-labels.idx1-ubyte"

	fmt.Println("Loading MNIST test data...")
	testData, err := datasets.LoadMNIST(testImagePath, testLabelPath, 100)
	if err != nil {
		log.Fatalf("Failed to load MNIST test data: %v", err)
	}

	fmt.Println("Loading model")
	model, err := nn.LoadModel("./models/model")
	if err != nil {
		log.Fatal("Error loading model: ", err)
	}

	fmt.Println("Evaluating the model...")
	accuracy := evaluate(model, testData.Images, testData.Labels)
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)
}
