% test_ferromode.m — MATLAB/Octave test suite for the Ferromode MEX bindings.
%
% Requires the built MEX file (ferromode_mex.{mex,mexa64,...}) and the m/
% wrappers on the path. Run: test_ferromode  (errors on any failure, so it
% can gate CI via a nonzero exit).
function test_ferromode
    fprintf('=== Ferromode MEX Test Suite ===\n');
    n = 200;
    t = linspace(0, 1, n);
    signal = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);

    % --- univariate algorithms ---
    r = ferromode_emd(signal, 'MaxIMFs', 4);
    assert(isstruct(r) && r.n_imfs >= 1, 'emd');
    assert(strcmp(r.algorithm, 'EMD'), 'emd algo');
    assert(size(r.imfs, 2) == n, 'emd imf width');

    % full config + palindrome boundary + spline
    r = ferromode_emd(signal, 'MaxIMFs', 4, 'BoundaryCondition', 'palindrome_cyclic', ...
                      'SplineType', 'periodic', 'EnergyThreshold', 1e-7);
    assert(r.n_imfs >= 1, 'emd palindrome');

    r = ferromode_eemd(signal, 'NumEnsembles', 5, 'NoiseStd', 0.2, 'Seed', 42);
    assert(strcmp(r.algorithm, 'EEMD'), 'eemd');
    r = ferromode_ceemd(signal, 'NumEnsembles', 5);
    assert(strcmp(r.algorithm, 'CEEMD'), 'ceemd');
    r = ferromode_ceemdan(signal, 'NumEnsembles', 5, 'MaxIMFs', 4);
    assert(strcmp(r.algorithm, 'CEEMDAN'), 'ceemdan');
    r = ferromode_iceemdan(signal, 'NumEnsembles', 5);
    assert(strcmp(r.algorithm, 'ICEEMDAN'), 'iceemdan');
    r = ferromode_vmd(signal, 'NModes', 2, 'Tau', 0.1);
    assert(strcmp(r.algorithm, 'VMD'), 'vmd');

    % --- multivariate (channels as columns) ---
    ch = [sin(2*pi*5*t)', cos(2*pi*5*t)'];
    r = ferromode_memd(ch, 'NumDirections', 16, 'MaxIMFs', 3);
    assert(strcmp(r.algorithm, 'MEMD'), 'memd');
    r = ferromode_namemd(ch, 'NumDirections', 16, 'MaxIMFs', 3, 'NNoiseChannels', 2, 'Seed', 42);
    assert(strcmp(r.algorithm, 'NA-MEMD'), 'namemd');

    % --- hilbert + reconstruct ---
    r = ferromode_emd(signal, 'MaxIMFs', 4);
    h = ferromode_hilbert(r.imfs, 100.0);
    assert(isfield(h, 'marginal_spectrum') && numel(h.marginal_spectrum) > 0, 'hilbert');
    rec = ferromode_reconstruct(r);
    assert(numel(rec) == n, 'reconstruct');

    % --- differentiable ---
    f = ferromode_emd_forward(signal, 'MaxIMFs', 4);
    assert(size(f.imfs, 1) >= 1 && isfinite(f.reconstruction_error), 'forward');
    g = ferromode_emd_backward(f.handle, ones(size(f.imfs)));
    assert(numel(g) == n && all(isfinite(g)), 'backward');

    % --- streaming ---
    hd = ferromode_streaming_new('MaxIMFs', 4, 'BufferSize', 2048, 'ArOrder', 3);
    s = ferromode_streaming_decompose_chunk(hd, sin(2*pi*5*linspace(0,1,256)));
    assert(size(s.imfs, 1) >= 1 && isfield(s, 'spectral_entropy'), 'streaming');
    ferromode_streaming_reset(hd);
    ferromode_streaming_free(hd);

    fprintf('ALL MEX TESTS PASSED\n');
end
