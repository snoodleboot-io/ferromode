% parse_memd_config.m — Parse name-value pair arguments into MEMD config struct.
function config = parse_memd_config(varargin)
    config = struct();
    for i = 1:2:length(varargin)
        key = varargin{i};
        val = varargin{i+1};
        switch lower(key)
            case 'numdirections'
                config.NumDirections = val;
            case 'maximfs'
                config.MaxIMFs = val;
            case 'sdthreshold'
                config.SDThreshold = val;
            case 'snumber'
                config.SNumber = val;
            case 'maxsiftingiterations'
                config.MaxSiftingIterations = val;
        end
    end
end
