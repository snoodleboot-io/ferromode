// Basic ferromode-go example: decompose a synthetic signal with EMD.
//
// Build:
//
//	cargo build --release -p ferromode-go
//	export CGO_LDFLAGS="-L$(pwd)/target/release -lferromode_go -Wl,-rpath,$(pwd)/target/release"
//	cd golang/examples/basic && go run main.go
package main

import (
	"fmt"
	"math"

	ferromode "github.com/ferromode/ferromode-go"
)

func main() {
	const n = 512
	const sampleRate = 256.0

	signal := make([]float64, n)
	for i := range signal {
		t := float64(i) / sampleRate
		signal[i] = math.Sin(2*math.Pi*32*t) +
			math.Sin(2*math.Pi*8*t) +
			math.Sin(2*math.Pi*2*t)
	}

	cfg := ferromode.DefaultEmdConfig()
	result, err := ferromode.EMD(signal, cfg)
	if err != nil {
		panic(err)
	}
	defer result.Free()

	fmt.Printf("Algorithm : %d\n", result.Algorithm)
	fmt.Printf("IMFs      : %d\n", result.NImfs)
	fmt.Printf("Samples   : %d\n", result.NSamples)
	fmt.Printf("Elapsed   : %.2f ms\n", result.ElapsedMs)

	reconstructed, err := result.Reconstruct()
	if err != nil {
		panic(err)
	}

	var maxErr float64
	for i, v := range reconstructed {
		if d := math.Abs(v - signal[i]); d > maxErr {
			maxErr = d
		}
	}
	fmt.Printf("Max reconstruction error: %.2e\n", maxErr)

	for i := 0; i < result.NImfs; i++ {
		imf, err := result.IMF(i)
		if err != nil {
			panic(err)
		}
		var energy float64
		for _, v := range imf {
			energy += v * v
		}
		fmt.Printf("IMF[%d] energy: %.4f\n", i, energy)
	}

	residue, err := result.Residue()
	if err != nil {
		panic(err)
	}
	var resEnergy float64
	for _, v := range residue {
		resEnergy += v * v
	}
	fmt.Printf("Residue energy: %.4f\n", resEnergy)
}
