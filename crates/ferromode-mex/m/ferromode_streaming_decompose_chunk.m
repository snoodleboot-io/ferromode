function result = ferromode_streaming_decompose_chunk(handle, chunk)
%FERROMODE_STREAMING_DECOMPOSE_CHUNK Decompose one chunk via a streaming handle.
%   RESULT has fields imfs, residue, spectral_entropy, stationarity_score,
%   extrema_spacing_cv.
    result = ferromode_mex('streaming_decompose_chunk', handle, chunk(:)');
end
