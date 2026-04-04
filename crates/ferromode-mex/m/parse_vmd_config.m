% parse_vmd_config.m — Parse name-value pair arguments into VMD config struct.
function config = parse_vmd_config(varargin)
    config = struct();
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch lower(key)
            case 'nmodes'
                config.NModes = val;
            case 'alpha'
                config.Alpha = val;
            case 'tau'
                config.Tau = val;
            case 'tol'
                config.Tol = val;
            case 'maxiterations'
                config.MaxIterations = val;
        end
    end
end
