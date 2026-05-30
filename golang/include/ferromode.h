#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Config structs (must match crates/ferromode/src/ffi.rs exactly)
 * ---------------------------------------------------------------------------*/

typedef struct {
    size_t   max_imfs;
    double   sd_threshold;
    size_t   s_number;
    size_t   max_sifting_iterations;
    int32_t  boundary_condition; /* 0=MirrorEven 1=MirrorOdd 2=Periodic 3=Slope
                                    4=ARModel 5=CharacteristicWave 6=WaveformMatching */
} CEmdConfig;

typedef struct {
    size_t   num_ensembles;
    double   noise_std;
    uint64_t seed;
    int32_t  use_seed;
} CEnsembleConfig;

typedef struct {
    size_t   num_directions;
    uint64_t direction_seed;
    size_t   max_imfs;
    double   sd_threshold;
    size_t   s_number;
    size_t   max_sifting_iterations;
} CMemdConfig;

typedef struct {
    CMemdConfig base;
    size_t      n_noise_channels;
    double      noise_std;
    uint64_t    seed;
    int32_t     use_seed;
} CNaMemdConfig;

typedef struct {
    size_t n_modes;
    double alpha;
    double tau;
    double tol;
    size_t max_iterations;
} CVmdConfig;

/* ---------------------------------------------------------------------------
 * Opaque result handle — never dereference from Go; use accessor functions.
 * ---------------------------------------------------------------------------*/
typedef struct CDecompositionResult CDecompositionResult;

/* ---------------------------------------------------------------------------
 * Algorithm entry points
 * ---------------------------------------------------------------------------*/
CDecompositionResult* ferromode_emd(
    const double* signal, size_t n_samples, const CEmdConfig* config);

CDecompositionResult* ferromode_eemd(
    const double* signal, size_t n_samples,
    const CEnsembleConfig* ensemble_config, const CEmdConfig* emd_config);

CDecompositionResult* ferromode_ceemd(
    const double* signal, size_t n_samples,
    const CEnsembleConfig* ensemble_config, const CEmdConfig* emd_config);

CDecompositionResult* ferromode_ceemdan(
    const double* signal, size_t n_samples,
    const CEnsembleConfig* ensemble_config, const CEmdConfig* emd_config);

CDecompositionResult* ferromode_iceemdan(
    const double* signal, size_t n_samples,
    const CEnsembleConfig* ensemble_config, const CEmdConfig* emd_config);

CDecompositionResult* ferromode_vmd(
    const double* signal, size_t n_samples, const CVmdConfig* config);

CDecompositionResult* ferromode_memd(
    const double* signal, size_t n_channels, size_t n_samples,
    const CMemdConfig* config);

CDecompositionResult* ferromode_namemd(
    const double* signal, size_t n_channels, size_t n_samples,
    const CNaMemdConfig* config);

/* ---------------------------------------------------------------------------
 * Result accessors
 * ---------------------------------------------------------------------------*/
size_t  ferromode_result_n_imfs(const CDecompositionResult* result);
size_t  ferromode_result_n_samples(const CDecompositionResult* result);
double  ferromode_result_elapsed_ms(const CDecompositionResult* result);
int32_t ferromode_result_algorithm(const CDecompositionResult* result);
int32_t ferromode_result_has_error(const CDecompositionResult* result);
const char* ferromode_result_error_msg(const CDecompositionResult* result);

int32_t ferromode_copy_imf(
    const CDecompositionResult* result,
    size_t imf_index, double* out, size_t n_samples);

int32_t ferromode_copy_residue(
    const CDecompositionResult* result, double* out, size_t n_samples);

int32_t ferromode_reconstruct(
    const CDecompositionResult* result, double* out, size_t n_samples);

/* ---------------------------------------------------------------------------
 * Memory management — caller must free every result exactly once.
 * ---------------------------------------------------------------------------*/
void ferromode_free_result(CDecompositionResult* result);

#ifdef __cplusplus
}
#endif
