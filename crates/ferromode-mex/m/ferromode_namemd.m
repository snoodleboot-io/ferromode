function result = ferromode_namemd(signal, varargin)
%FERROMODE_NAMEMD Noise-Assisted Multivariate Empirical Mode Decomposition.
%
%   RESULT = FERROMODE_NAMEMD(SIGNAL) decomposes a multivariate SIGNAL using
%   the NA-MEMD algorithm. NA-MEMD adds white noise channels to improve
%   mode alignment across channels and reduce mode mixing.
%
%   SIGNAL should be an n_samples x n_channels matrix where each column
%   is a channel.
%
%   RESULT = FERROMODE_NAMEMD(SIGNAL, Name, Value, ...) specifies optional
%   name-value pair arguments:
%
%   'NNoiseChannels'        Number of noise channels (default: 2)
%   'NoiseStd'              Noise std fraction (default: 0.1)
%   'Seed'                  Random seed (default: [])
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
%     result = ferromode_namemd(signal, 'NNoiseChannels', 3, 'Seed', 42);
%
%   See also FERROMODE_MEMD.

    config = parse_namemd_config(varargin{:});
    result = ferromode_mex('namemd', signal, config);
end
