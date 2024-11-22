package datasets

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
)

type MNISTData struct {
	Images [][]float64
	Labels [][]float64
}

func LoadMNIST(imagePath, labelPath string, numSamples int) (*MNISTData, error) {
	images, err := loadImages(imagePath, numSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to load images: %w", err)
	}

	labels, err := loadLabels(labelPath, numSamples)
	if err != nil {
		return nil, fmt.Errorf("failed to load labels: %w", err)
	}

	return &MNISTData{
		Images: images,
		Labels: labels,
	}, nil
}

func loadImages(path string, numSamples int) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open image file: %w", err)
	}
	defer file.Close()

	var magicNumber, numImages, numRows, numCols int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if magicNumber != 2051 {
		return nil, errors.New("invalid image file magic number")
	}

	if err := binary.Read(file, binary.BigEndian, &numImages); err != nil {
		return nil, fmt.Errorf("failed to read number of images: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &numRows); err != nil {
		return nil, fmt.Errorf("failed to read number of rows: %w", err)
	}
	if err := binary.Read(file, binary.BigEndian, &numCols); err != nil {
		return nil, fmt.Errorf("failed to read number of columns: %w", err)
	}

	if numSamples > int(numImages) {
		return nil, errors.New("requested number of samples exceeds available images")
	}

	imageSize := int(numRows * numCols)
	images := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		buf := make([]byte, imageSize)
		if _, err := file.Read(buf); err != nil {
			return nil, fmt.Errorf("failed to read image data: %w", err)
		}

		image := make([]float64, imageSize)
		for j, pixel := range buf {
			image[j] = float64(pixel) / 255.0
		}
		images[i] = image
	}

	return images, nil
}

func loadLabels(path string, numSamples int) ([][]float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open label file: %w", err)
	}
	defer file.Close()

	var magicNumber, numLabels int32
	if err := binary.Read(file, binary.BigEndian, &magicNumber); err != nil {
		return nil, fmt.Errorf("failed to read magic number: %w", err)
	}
	if magicNumber != 2049 {
		return nil, errors.New("invalid label file magic number")
	}

	if err := binary.Read(file, binary.BigEndian, &numLabels); err != nil {
		return nil, fmt.Errorf("failed to read number of labels: %w", err)
	}

	if numSamples > int(numLabels) {
		return nil, errors.New("requested number of samples exceeds available labels")
	}

	labels := make([][]float64, numSamples)
	for i := 0; i < numSamples; i++ {
		var label byte
		if err := binary.Read(file, binary.BigEndian, &label); err != nil {
			return nil, fmt.Errorf("failed to read label data: %w", err)
		}

		oneHot := make([]float64, 10)
		oneHot[label] = 1.0
		labels[i] = oneHot
	}

	return labels, nil
}
