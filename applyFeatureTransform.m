function transformed = applyFeatureTransform(data, state)
%APPLYFEATURETRANSFORM Apply frozen training parameters, preserving labels.
validateattributes(data, {'numeric'}, {'2d','nonempty','finite'});
if size(data,2) ~= state.inputFeatures+1
    error('Research:InvalidFeatures', 'Predictor count differs from training.');
end
Z = (data(:,1:end-1)-state.mu)./state.sigma;
switch state.method
    case 'none'
        transformed = data;
    case 'pca'
        transformed = [(Z-state.center)*state.coeff, data(:,end)];
    case 'nca'
        transformed = [Z(:,state.indices), data(:,end)];
    otherwise
        error('Research:InvalidFeatures', 'Unknown fitted feature method.');
end
end
