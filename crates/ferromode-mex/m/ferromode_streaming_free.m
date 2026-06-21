function ferromode_streaming_free(handle)
%FERROMODE_STREAMING_FREE Release a streaming handle.
    ferromode_mex('streaming_free', handle);
end
