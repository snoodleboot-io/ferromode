function result = ferromode_vmd(signal, varargin)
%FERROMODE_VMD Variational Mode Decomposition.
%
%   RESULT = FERROMODE_VMD(SIGNAL) decomposes SIGNAL using the VMD algorithm.
%   VMD solves a variational optimization problem to extract band-limited
%   modes, unlike EMD which is iterative.
%
%   RESULT = FERROMODE_VMD(SIGNAL, Name, Value, ...) specifies optional
%   name-value pair arguments:
%
%   'NModes'            Number of modes to extract (default: 3)
%   'Alpha'             Bandwidth penalty (default: 2000)
%   'Tau'               Dual ascent step size (default: 0)
%   'Tol'               Convergence tolerance (default: 1e-7)
%   'MaxIterations'     Maximum ADMM iterations (default: 500)
%
%   RESULT is a struct with fields: imfs, residue, n_imfs, algorithm,
%   elapsed_ms, n_siftings.
%
%   Example:
%     t = linspace(0, 1, 1000);
%     x = sin(2*pi*5*t) + 0.5*sin(2*pi*20*t);
%     result = ferromode_vmd(x, 'NModes', 2, 'Alpha', 2000);
%
%   See also FERROMODE_EMD, FERROMODE_EEMD.

    config = parse_vmd_config(varargin{:});
    result = ferromode_mex('vmd', signal(:)', config);
end
