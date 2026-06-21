function grad = ferromode_emd_backward(grad_imfs, signal)
%FERROMODE_EMD_BACKWARD Differentiable EMD backward pass (placeholder).
%   GRAD = FERROMODE_EMD_BACKWARD(GRAD_IMFS, SIGNAL) where GRAD_IMFS is an
%   (n_imfs x n_samples) matrix of upstream gradients. Returns the gradient
%   w.r.t. the input signal (currently the average of upstream gradients).
    grad = ferromode_mex('emd_backward', grad_imfs, signal(:)');
end
