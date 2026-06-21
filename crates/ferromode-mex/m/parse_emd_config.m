% parse_emd_config.m — Parse name-value pair arguments into EMD config struct.
function config = parse_emd_config(varargin)
    config = struct();
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch lower(key)
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
            case 'splinetype'
                config.SplineType = val;
            case 'fixediterations'
                config.FixedIterations = val;
            case 'energythreshold'
                config.EnergyThreshold = val;
            case 'reconstructiontolerance'
                config.ReconstructionTolerance = val;
            case 'validatereconstruction'
                config.ValidateReconstruction = double(logical(val));
            case 'intermittencycv'
                config.IntermittencyCV = val;
            case 'intermittencyminintervals'
                config.IntermittencyMinIntervals = val;
        end
    end
end
