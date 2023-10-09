package main

import (
	"fmt"
	// "runtime"

	// "github.com/Applifier/go-xgboost"
	"github.com/dmitryikh/leaves"
)

func main() {
	// loading model
	model, err := leaves.XGEnsembleFromFile("./xgdermatology.model", false)
	if err != nil {
		fmt.Print(err)
	}
	fmt.Print(model)
	fmt.Printf("Name: %s\n", model.Name())
	fmt.Printf("NFeatures: %d\n", model.NFeatures())
	fmt.Printf("NOutputGroups: %d\n", model.NOutputGroups())
	fmt.Printf("NEstimators: %d\n", model.NEstimators())
	fvals := []float64{1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0}
	p := model.PredictSingle(fvals, 0)
	fmt.Printf("Prediction for %v: %f\n", fvals, p)
	// predictor, _ := xgboost.NewPredictor("/Users/apple/WorkSpace/goWork/demo/bin_model", runtime.NumCPU(), 0, 0, -1)
	// res, _ := predictor.Predict(xgboost.FloatSliceVector([]float32{1, 2, 3}))
	// fmt.Printf("Results: %+v\n", predictor)

}

// - xgboost -> python[version='>=3.10,<3.11.0a0|>=3.11,<3.12.0a0|>=3.9,<3.10.0a0|>=3.8,<3.9.0a0|>=3.7,<3.8.0a0|>=3.6,<3.7.0a0']
