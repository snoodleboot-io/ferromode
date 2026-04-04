function result = ferromode_emd(signal, varargin)
%FERROMODE_EMD Empirical Mode Decomposition.
%
%   RESULT = FERROMODE_EMD(SIGNAL) decomposes SIGNAL using the EMD algorithm.
%
%   RESULT = FERROMODE_EMD(SIGNAL, Name, Value, ...) specifies optional
%   name-value pair arguments to configure the decomposition:
%
%   'MaxIMFs'             Maximum number of IMFs to extract (default: 0 = auto)
%   'SDThreshold'         Stopping criterion SD threshold (default: 0.2)
%   'SNumber'             S-number for stopping criterion (default: 5)
%   'MaxSiftingIterations' Maximum sifting iterations (default: 100)
%   'BoundaryCondition'   Boundary handling: 'mirror', 'periodic', 'slope',
%                         'ar', 'characteristic', 'waveform' (default: 'mirror')
%
%   RESULT is a struct with fields:
%     imfs       - n_imfs x n_samples matrix of IMFs
%     residue    - 1 x n_samples residue vector
%     n_imfs     - number of IMFs extracted
%     algorithm  - algorithm name ('EMD')
%     elapsed_ms - computation time in milliseconds
%     n_siftings - total sifting iterations
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);
%     result = ferromode_emd(x);
%     plot(result.imfs');
%
%   See also FERROMODE_EEMD, FERROMODE_CEEMD, FERROMODE_CEEMDAN,
%   FERROMODE_ICEEMDAN, FERROMODE_VMD, FERROMODE_MEMD, FERROMODE_NAMEMD.

    config = parse_emd_config(varargin{:});
    result = ferromode_mex('emd', signal(:)', config);
end
