function handle = ferromode_streaming_new(varargin)
%FERROMODE_STREAMING_NEW Create a streaming decomposer handle.
%   HANDLE = FERROMODE_STREAMING_NEW(Name, Value, ...) accepts EMD config
%   options plus 'BufferSize' and 'ArOrder'. Returns an opaque numeric handle.
    buffer_size = 4096; ar_order = 3; cfgargs = {};
    i = 1;
    while i <= numel(varargin)
        key = varargin{i};
        if strcmpi(key, 'BufferSize'), buffer_size = varargin{i+1};
        elseif strcmpi(key, 'ArOrder'), ar_order = varargin{i+1};
        else, cfgargs(end+1:end+2) = varargin(i:i+1); end
        i = i + 2;
    end
    config = parse_emd_config(cfgargs{:});
    handle = ferromode_mex('streaming_new', config, buffer_size, ar_order);
end
