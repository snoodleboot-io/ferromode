function result = ferromode_iceemdan(signal, varargin)
%FERROMODE_ICEEMDAN Improved Complete Ensemble EMD with Adaptive Noise.
%
%   RESULT = FERROMODE_ICEEMDAN(SIGNAL) decomposes SIGNAL using the ICEEMDAN
%   algorithm. ICEEMDAN uses the k-th IMF of a noise-only signal instead of
%   raw white noise at each stage, producing cleaner IMFs with less residual
%   noise.
%
%   RESULT = FERROMODE_ICEEMDAN(SIGNAL, Name, Value, ...) accepts the same
%   name-value pairs as FERROMODE_EEMD.
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);
%     result = ferromode_iceemdan(x, 'NumEnsembles', 50, 'Seed', 42);
%
%   See also FERROMODE_CEEMDAN, FERROMODE_CEEMD.

    config = parse_eemd_config(varargin{:});
    result = ferromode_mex('iceemdan', signal(:)', config);
end
