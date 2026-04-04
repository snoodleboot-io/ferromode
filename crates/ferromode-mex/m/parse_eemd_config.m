% parse_eemd_config.m — Parse name-value pair arguments into EEMD config struct.
function config = parse_eemd_config(varargin)
    config = struct();
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch lower(key)
            case 'numensembles'
                config.NumEnsembles = val;
            case 'noisestd'
                config.NoiseStd = val;
            case 'seed'
                config.Seed = val;
            case 'maximfs'
                config.MaxIMFs = val;
            case 'sdthreshold'
                config.SDThreshold = val;
            case 'snumber'
                config.SNumber = val;
            case 'maxsiftingiterations'
                config.MaxSiftingIterations = val;
            case 'boundarycondition'
                config.BoundaryCondition = val;
        end
    end
end
