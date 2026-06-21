function signal = ferromode_reconstruct(result)
%FERROMODE_RECONSTRUCT Sum a decomposition result's IMFs and residue.
    signal = ferromode_mex('reconstruct', result);
end
