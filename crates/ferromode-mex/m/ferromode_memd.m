function result = ferromode_memd(signal, varargin)
%FERROMODE_MEMD Multivariate Empirical Mode Decomposition.
%
%   RESULT = FERROMODE_MEMD(SIGNAL) decomposes a multivariate SIGNAL using
%   the MEMD algorithm. SIGNAL should be an n_samples x n_channels matrix
%   where each column is a channel.
%
%   RESULT = FERROMODE_MEMD(SIGNAL, Name, Value, ...) specifies optional
%   name-value pair arguments:
%
%   'NumDirections'         Number of direction vectors (default: 16)
%   'MaxIMFs'               Maximum number of IMFs (default: 0 = auto)
%   'SDThreshold'           SD threshold (default: 0.2)
%   'SNumber'               S-number (default: 5)
%   'MaxSiftingIterations'  Max sifting iterations (default: 100)
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 500);
%     ch1 = sin(2*pi*5*t);
%     ch2 = sin(2*pi*5*t + pi/4);
%     signal = [ch1', ch2'];
%     result = ferromode_memd(signal, 'NumDirections', 32);
%
%   See also FERROMODE_NAMEMD.

    config = parse_memd_config(varargin{:});
    result = ferromode_mex('memd', signal, config);
end
