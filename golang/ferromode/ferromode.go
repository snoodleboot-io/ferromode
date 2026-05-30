// Package ferromode provides Go bindings for the Ferromode EMD library.
//
// The library calls into a native Rust shared library (libferromode_go.so)
// via cgo. Build the shared library first:
//
//	cargo build --release -p ferromode-go
//
// Then set the linker flags to point at the build output:
//
//	export CGO_LDFLAGS="-L/path/to/ferromode/target/release -lferromode_go -Wl,-rpath,/path/to/ferromode/target/release"
//
// Pure-wrap contract: this package is marshalling only — no algorithm logic.
package ferromode

/*
#cgo CFLAGS: -I${SRCDIR}/../../include
#include "ferromode.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

// EmdConfig holds configuration for EMD and single-signal decompositions.
type EmdConfig struct {
	MaxImfs              int
	SdThreshold          float64
	SNumber              int
	MaxSiftingIterations int
	// BoundaryCondition: 0=MirrorEven 1=MirrorOdd 2=Periodic 3=Slope
	//                    4=ARModel 5=CharacteristicWave 6=WaveformMatching
	BoundaryCondition int
}

// DefaultEmdConfig returns a config matching Ferromode defaults.
func DefaultEmdConfig() EmdConfig {
	return EmdConfig{
		MaxImfs:              0,
		SdThreshold:          0.2,
		SNumber:              5,
		MaxSiftingIterations: 100,
		BoundaryCondition:    0,
	}
}

// EnsembleConfig holds configuration for ensemble methods (EEMD, CEEMDAN, ICEEMDAN).
type EnsembleConfig struct {
	NumEnsembles int
	NoiseStd     float64
	Seed         uint64
	UseSeed      bool
}

// DefaultEnsembleConfig returns sensible ensemble defaults.
func DefaultEnsembleConfig() EnsembleConfig {
	return EnsembleConfig{NumEnsembles: 100, NoiseStd: 0.2}
}

// VmdConfig holds configuration for Variational Mode Decomposition.
type VmdConfig struct {
	NModes        int
	Alpha         float64
	Tau           float64
	Tol           float64
	MaxIterations int
}

// DefaultVmdConfig returns VMD defaults.
func DefaultVmdConfig() VmdConfig {
	return VmdConfig{NModes: 3, Alpha: 2000.0, Tau: 0.0, Tol: 1e-7, MaxIterations: 500}
}

// MemdConfig holds configuration for Multivariate EMD.
type MemdConfig struct {
	NumDirections        int
	DirectionSeed        uint64
	MaxImfs              int
	SdThreshold          float64
	SNumber              int
	MaxSiftingIterations int
}

// DefaultMemdConfig returns MEMD defaults.
func DefaultMemdConfig() MemdConfig {
	return MemdConfig{NumDirections: 64, SdThreshold: 0.2, SNumber: 5, MaxSiftingIterations: 100}
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// Result holds the output of a decomposition. Call Free when done.
type Result struct {
	ptr       *C.CDecompositionResult
	NImfs     int
	NSamples  int
	ElapsedMs float64
	Algorithm int
}

func newResult(ptr *C.CDecompositionResult) (*Result, error) {
	if ptr == nil {
		return nil, fmt.Errorf("ferromode: null result pointer")
	}
	if C.ferromode_result_has_error(ptr) != 0 {
		msg := C.GoString(C.ferromode_result_error_msg(ptr))
		C.ferromode_free_result(ptr)
		return nil, fmt.Errorf("ferromode: %s", msg)
	}
	r := &Result{
		ptr:       ptr,
		NImfs:     int(C.ferromode_result_n_imfs(ptr)),
		NSamples:  int(C.ferromode_result_n_samples(ptr)),
		ElapsedMs: float64(C.ferromode_result_elapsed_ms(ptr)),
		Algorithm: int(C.ferromode_result_algorithm(ptr)),
	}
	runtime.SetFinalizer(r, (*Result).Free)
	return r, nil
}

// IMF returns the i-th Intrinsic Mode Function (0-indexed).
func (r *Result) IMF(i int) ([]float64, error) {
	if i < 0 || i >= r.NImfs {
		return nil, fmt.Errorf("ferromode: IMF index %d out of range [0, %d)", i, r.NImfs)
	}
	buf := make([]float64, r.NSamples)
	rc := C.ferromode_copy_imf(r.ptr, C.size_t(i),
		(*C.double)(unsafe.Pointer(&buf[0])), C.size_t(r.NSamples))
	if rc != 0 {
		return nil, fmt.Errorf("ferromode: ferromode_copy_imf returned %d", rc)
	}
	return buf, nil
}

// Residue returns the final residue signal.
func (r *Result) Residue() ([]float64, error) {
	buf := make([]float64, r.NSamples)
	rc := C.ferromode_copy_residue(r.ptr,
		(*C.double)(unsafe.Pointer(&buf[0])), C.size_t(r.NSamples))
	if rc != 0 {
		return nil, fmt.Errorf("ferromode: ferromode_copy_residue returned %d", rc)
	}
	return buf, nil
}

// Reconstruct returns the sum of all IMFs and the residue (≈ original signal).
func (r *Result) Reconstruct() ([]float64, error) {
	buf := make([]float64, r.NSamples)
	rc := C.ferromode_reconstruct(r.ptr,
		(*C.double)(unsafe.Pointer(&buf[0])), C.size_t(r.NSamples))
	if rc != 0 {
		return nil, fmt.Errorf("ferromode: ferromode_reconstruct returned %d", rc)
	}
	return buf, nil
}

// Free releases the underlying Rust memory. Safe to call more than once.
func (r *Result) Free() {
	if r.ptr != nil {
		C.ferromode_free_result(r.ptr)
		r.ptr = nil
	}
}

// ---------------------------------------------------------------------------
// cgo config converters
// ---------------------------------------------------------------------------

func toCEmdConfig(c EmdConfig) C.CEmdConfig {
	return C.CEmdConfig{
		max_imfs:               C.size_t(c.MaxImfs),
		sd_threshold:           C.double(c.SdThreshold),
		s_number:               C.size_t(c.SNumber),
		max_sifting_iterations: C.size_t(c.MaxSiftingIterations),
		boundary_condition:     C.int32_t(c.BoundaryCondition),
	}
}

func toCEnsembleConfig(c EnsembleConfig) C.CEnsembleConfig {
	useSeed := C.int32_t(0)
	if c.UseSeed {
		useSeed = 1
	}
	return C.CEnsembleConfig{
		num_ensembles: C.size_t(c.NumEnsembles),
		noise_std:     C.double(c.NoiseStd),
		seed:          C.uint64_t(c.Seed),
		use_seed:      useSeed,
	}
}

func toCVmdConfig(c VmdConfig) C.CVmdConfig {
	return C.CVmdConfig{
		n_modes:        C.size_t(c.NModes),
		alpha:          C.double(c.Alpha),
		tau:            C.double(c.Tau),
		tol:            C.double(c.Tol),
		max_iterations: C.size_t(c.MaxIterations),
	}
}

func toCMemdConfig(c MemdConfig) C.CMemdConfig {
	return C.CMemdConfig{
		num_directions:         C.size_t(c.NumDirections),
		direction_seed:         C.uint64_t(c.DirectionSeed),
		max_imfs:               C.size_t(c.MaxImfs),
		sd_threshold:           C.double(c.SdThreshold),
		s_number:               C.size_t(c.SNumber),
		max_sifting_iterations: C.size_t(c.MaxSiftingIterations),
	}
}

// signalPtr returns a stable C pointer into a []float64 slice.
// The slice must remain alive for the duration of the C call.
func signalPtr(s []float64) (*C.double, C.size_t) {
	return (*C.double)(unsafe.Pointer(&s[0])), C.size_t(len(s))
}

// ---------------------------------------------------------------------------
// Algorithm functions
// ---------------------------------------------------------------------------

// EMD decomposes signal using Empirical Mode Decomposition.
func EMD(signal []float64, config EmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	cfg := toCEmdConfig(config)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_emd(ptr, n, &cfg))
}

// EEMD decomposes signal using Ensemble EMD.
func EEMD(signal []float64, ensemble EnsembleConfig, emd EmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	ens := toCEnsembleConfig(ensemble)
	ec := toCEmdConfig(emd)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_eemd(ptr, n, &ens, &ec))
}

// CEEMD decomposes signal using Complete Ensemble EMD.
func CEEMD(signal []float64, ensemble EnsembleConfig, emd EmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	ens := toCEnsembleConfig(ensemble)
	ec := toCEmdConfig(emd)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_ceemd(ptr, n, &ens, &ec))
}

// CEEMDAN decomposes signal using CEEMDAN.
func CEEMDAN(signal []float64, ensemble EnsembleConfig, emd EmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	ens := toCEnsembleConfig(ensemble)
	ec := toCEmdConfig(emd)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_ceemdan(ptr, n, &ens, &ec))
}

// ICEEMDAN decomposes signal using Improved CEEMDAN.
func ICEEMDAN(signal []float64, ensemble EnsembleConfig, emd EmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	ens := toCEnsembleConfig(ensemble)
	ec := toCEmdConfig(emd)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_iceemdan(ptr, n, &ens, &ec))
}

// VMD decomposes signal using Variational Mode Decomposition.
func VMD(signal []float64, config VmdConfig) (*Result, error) {
	if len(signal) == 0 {
		return nil, fmt.Errorf("ferromode: signal must not be empty")
	}
	cfg := toCVmdConfig(config)
	ptr, n := signalPtr(signal)
	return newResult(C.ferromode_vmd(ptr, n, &cfg))
}

// MEMD decomposes a multivariate signal (channels × samples matrix, row-major).
// signal must have length nChannels * nSamples.
func MEMD(signal []float64, nChannels int, config MemdConfig) (*Result, error) {
	if len(signal) == 0 || nChannels <= 0 || len(signal)%nChannels != 0 {
		return nil, fmt.Errorf("ferromode: invalid MEMD input: len=%d nChannels=%d", len(signal), nChannels)
	}
	nSamples := len(signal) / nChannels
	cfg := toCMemdConfig(config)
	ptr := (*C.double)(unsafe.Pointer(&signal[0]))
	return newResult(C.ferromode_memd(ptr, C.size_t(nChannels), C.size_t(nSamples), &cfg))
}
