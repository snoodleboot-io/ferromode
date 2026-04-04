function result = ferromode_ceemdan(signal, varargin)
%FERROMODE_CEEMDAN Complete Ensemble EMD with Adaptive Noise.
%
%   RESULT = FERROMODE_CEEMDAN(SIGNAL) decomposes SIGNAL using the CEEMDAN
%   algorithm. CEEMDAN adds noise stage-wise rather than to the original
%   signal, ensuring complete reconstruction and better spectral separation.
%
%   RESULT = FERROMODE_CEEMDAN(SIGNAL, Name, Value, ...) accepts the same
%   name-value pairs as FERROMODE_EEMD.
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);
%     result = ferromode_ceemdan(x, 'NumEnsembles', 50, 'Seed', 42);
%
%   See also FERROMODE_EEMD, FERROMODE_ICEEMDAN.

    config = parse_eemd_config(varargin{:});
    result = ferromode_mex('ceemdan', signal(:)', config);
end
