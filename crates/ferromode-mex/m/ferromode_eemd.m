function result = ferromode_eemd(signal, varargin)
%FERROMODE_EEMD Ensemble Empirical Mode Decomposition.
%
%   RESULT = FERROMODE_EEMD(SIGNAL) decomposes SIGNAL using the EEMD algorithm.
%
%   RESULT = FERROMODE_EEMD(SIGNAL, Name, Value, ...) specifies optional
%   name-value pair arguments:
%
%   'NumEnsembles'        Number of ensemble trials (default: 100)
%   'NoiseStd'            Noise standard deviation fraction (default: 0.2)
%   'Seed'                Random seed for reproducibility (default: [])
%   'MaxIMFs'             Maximum number of IMFs (default: 0 = auto)
%   'SDThreshold'         SD threshold (default: 0.2)
%   'SNumber'             S-number (default: 5)
%   'MaxSiftingIterations' Max sifting iterations (default: 100)
%   'BoundaryCondition'   Boundary handling (default: 'mirror')
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.1*sin(2*pi*50*t);
%     result = ferromode_eemd(x, 'NumEnsembles', 50, 'Seed', 42);
%
%   See also FERROMODE_EMD, FERROMODE_CEEMD, FERROMODE_CEEMDAN.

    config = parse_eemd_config(varargin{:});
    result = ferromode_mex('eemd', signal(:)', config);
end
