% test_ferromode.m — MATLAB test script for Ferromode MEX bindings.
%
% Run: test_ferromode
% Each test prints PASS or FAIL.

function test_ferromode
    fprintf('=== Ferromode MEX Test Suite ===\n\n');

    n = 200;
    t = linspace(0, 1, n);
    signal = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);

    % Test 1: EMD
    fprintf('Test 1: ferromode_emd... ');
    try
        result = ferromode_emd(signal);
        assert(isstruct(result), 'result must be struct');
        assert(isfield(result, 'imfs'), 'must have imfs field');
        assert(isfield(result, 'residue'), 'must have residue field');
        assert(isfield(result, 'n_imfs'), 'must have n_imfs field');
        assert(isfield(result, 'algorithm'), 'must have algorithm field');
        assert(isfield(result, 'elapsed_ms'), 'must have elapsed_ms field');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(size(result.imfs, 2) == n, 'IMFs must have n_samples columns');
        assert(strcmp(result.algorithm, 'EMD'), 'algorithm must be EMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 2: EEMD
    fprintf('Test 2: ferromode_eemd... ');
    try
        result = ferromode_eemd(signal, 'NumEnsembles', 10, 'Seed', 42);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'EEMD'), 'algorithm must be EEMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 3: CEEMD
    fprintf('Test 3: ferromode_ceemd... ');
    try
        result = ferromode_ceemd(signal, 'NumEnsembles', 10, 'Seed', 42);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'CEEMD'), 'algorithm must be CEEMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 4: CEEMDAN
    fprintf('Test 4: ferromode_ceemdan... ');
    try
        result = ferromode_ceemdan(signal, 'NumEnsembles', 10, 'Seed', 42);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'CEEMDAN'), 'algorithm must be CEEMDAN');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 5: ICEEMDAN
    fprintf('Test 5: ferromode_iceemdan... ');
    try
        result = ferromode_iceemdan(signal, 'NumEnsembles', 10, 'Seed', 42);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'ICEEMDAN'), 'algorithm must be ICEEMDAN');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 6: VMD
    fprintf('Test 6: ferromode_vmd... ');
    try
        result = ferromode_vmd(signal, 'NModes', 2);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'VMD'), 'algorithm must be VMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 7: MEMD (bivariate signal)
    fprintf('Test 7: ferromode_memd... ');
    try
        ch1 = sin(2*pi*5*t);
        ch2 = sin(2*pi*5*t + pi/4);
        mv_signal = [ch1', ch2'];
        result = ferromode_memd(mv_signal, 'NumDirections', 16);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'MEMD'), 'algorithm must be MEMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 8: NA-MEMD (bivariate signal)
    fprintf('Test 8: ferromode_namemd... ');
    try
        ch1 = sin(2*pi*5*t);
        ch2 = sin(2*pi*5*t + pi/4);
        mv_signal = [ch1', ch2'];
        result = ferromode_namemd(mv_signal, 'NNoiseChannels', 2, 'Seed', 42);
        assert(isstruct(result), 'result must be struct');
        assert(size(result.imfs, 1) >= 1, 'must have at least 1 IMF');
        assert(strcmp(result.algorithm, 'NA-MEMD'), 'algorithm must be NA-MEMD');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 9: Reconstruction
    fprintf('Test 9: ferromode_reconstruct... ');
    try
        result = ferromode_emd(signal);
        reconstructed = ferromode_reconstruct(result);
        assert(length(reconstructed) == n, 'reconstructed must have n samples');
        max_error = max(abs(signal - reconstructed));
        assert(max_error < 1e-6, sprintf('reconstruction error too large: %e', max_error));
        fprintf('PASS (error: %.2e)\n', max_error);
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 10: EMD with config
    fprintf('Test 10: ferromode_emd with config... ');
    try
        result = ferromode_emd(signal, 'MaxIMFs', 3, 'BoundaryCondition', 'periodic');
        assert(isstruct(result), 'result must be struct');
        assert(result.n_imfs <= 3, 'n_imfs must be <= MaxIMFs');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    % Test 11: VMD with config
    fprintf('Test 11: ferromode_vmd with config... ');
    try
        result = ferromode_vmd(signal, 'NModes', 2, 'Alpha', 2000, 'MaxIterations', 200);
        assert(isstruct(result), 'result must be struct');
        assert(result.n_imfs == 2, 'VMD with NModes=2 must produce 2 IMFs');
        fprintf('PASS\n');
    catch e
        fprintf('FAIL: %s\n', e.message);
    end

    fprintf('\n=== All tests completed ===\n');
end
