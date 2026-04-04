function result = ferromode_ceemd(signal, varargin)
%FERROMODE_CEEMD Complementary Ensemble Empirical Mode Decomposition.
%
%   RESULT = FERROMODE_CEEMD(SIGNAL) decomposes SIGNAL using the CEEMD algorithm.
%
%   CEEMD improves upon EEMD by using complementary noise pairs (+noise and
%   -noise) which cancel noise residue more effectively.
%
%   RESULT = FERROMODE_CEEMD(SIGNAL, Name, Value, ...) accepts the same
%   name-value pairs as FERROMODE_EEMD.
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.1*sin(2*pi*50*t);
%     result = ferromode_ceemd(x, 'NumEnsembles', 50, 'Seed', 42);
%
%   See also FERROMODE_EEMD, FERROMODE_CEEMDAN.

    config = parse_eemd_config(varargin{:});
    result = ferromode_mex('ceemd', signal(:)', config);
end
