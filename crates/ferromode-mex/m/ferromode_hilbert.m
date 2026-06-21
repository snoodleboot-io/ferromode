function result = ferromode_hilbert(imfs, sample_rate)
%FERROMODE_HILBERT Hilbert spectral analysis of a set of IMFs.
%   RESULT = FERROMODE_HILBERT(IMFS, SAMPLE_RATE) where IMFS is an
%   (n_imfs x n_samples) matrix. RESULT has fields instantaneous_amplitude,
%   instantaneous_frequency, marginal_spectrum.
    if nargin < 2, sample_rate = 1.0; end
    result = ferromode_mex('hilbert', imfs, sample_rate);
end
