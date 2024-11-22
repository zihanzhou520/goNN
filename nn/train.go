package nn

import (
	"errors"
	"fmt"
	"github.com/cheggaaa/pb/v3"
	"math/rand"
	"yashwanthrs.com/m/util"
)

func Train(model *Model, xTrain, yTrain util.Matrix, epochs int, learningRate float64, batchSize int) error {
	if len(xTrain) != len(yTrain) {
		return errors.New("xTrain and yTrain must have the same number of samples")
	}

	numSamples := len(xTrain)
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("Epoch %d/%d\n", epoch+1, epochs)

		xTrain, yTrain = Shuffle(xTrain, yTrain)

		progressBar := pb.StartNew(numSamples / batchSize)

		for i := 0; i < numSamples; i += batchSize {
			end := i + batchSize
			if end > numSamples {
				end = numSamples
			}

			xBatch := xTrain[i:end]
			yBatch := yTrain[i:end]

			output, err := model.Forward(xBatch)
			if err != nil {
				return fmt.Errorf("error during forward pass: %v", err)
			}

			_, dLoss := ComputeLoss(output, yBatch)

			err = model.Backward(dLoss)
			if err != nil {
				return fmt.Errorf("error during backward pass: %v", err)
			}

			model.UpdateWeights(learningRate)
			progressBar.Increment()
		}

		progressBar.Finish()

		epochLoss := ComputeTotalLoss(model, xTrain, yTrain)
		fmt.Printf("Epoch %d Loss: %.6f\n", epoch+1, epochLoss)
	}

	return nil
}

func ComputeLoss(predicted, actual util.Matrix) (float64, util.Matrix) {
	loss := 0.0
	dLoss := util.NewMatrix(len(predicted), len(predicted[0]))

	for i := range predicted {
		for j := range predicted[i] {
			diff := predicted[i][j] - actual[i][j]
			loss += diff * diff
			dLoss[i][j] = 2 * diff
		}
	}

	loss /= float64(len(predicted))
	return loss, dLoss
}

func ComputeTotalLoss(model *Model, x, y util.Matrix) float64 {
	output, err := model.Forward(x)
	if err != nil {
		panic(fmt.Sprintf("error during forward pass: %v", err))
	}

	loss, _ := ComputeLoss(output, y)
	return loss
}

func Shuffle(x, y util.Matrix) (util.Matrix, util.Matrix) {
	indices := rand.Perm(len(x))
	xShuffled := make(util.Matrix, len(x))
	yShuffled := make(util.Matrix, len(y))

	for i, idx := range indices {
		xShuffled[i] = x[idx]
		yShuffled[i] = y[idx]
	}

	return xShuffled, yShuffled
}
