function result = ferromode_emd_forward(signal, varargin)
%FERROMODE_EMD_FORWARD Differentiable EMD forward pass.
%   RESULT = FERROMODE_EMD_FORWARD(SIGNAL, Name, Value, ...) returns a struct
%   with fields imfs, residue, num_sifts, reconstruction_error, and handle.
%   Pass RESULT.handle to FERROMODE_EMD_BACKWARD.
    config = parse_emd_config(varargin{:});
    result = ferromode_mex('emd_forward', signal(:)', config);
end
