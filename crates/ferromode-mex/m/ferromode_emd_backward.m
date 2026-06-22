function grad = ferromode_emd_backward(handle, grad_imfs)
%FERROMODE_EMD_BACKWARD Differentiable EMD backward pass (exact VJP).
%   GRAD = FERROMODE_EMD_BACKWARD(HANDLE, GRAD_IMFS) where HANDLE is the
%   `handle` field returned by FERROMODE_EMD_FORWARD and GRAD_IMFS is an
%   (n_imfs x n_samples) matrix of upstream gradients (one per IMF). Returns the
%   exact gradient w.r.t. the input signal.
    grad = ferromode_mex('emd_backward', handle, grad_imfs);
end
