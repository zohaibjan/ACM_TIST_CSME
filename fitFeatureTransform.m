function state = fitFeatureTransform(trainData, params)
%FITFEATURETRANSFORM Fit scaling and optional PCA/NCA on training rows only.
% Feature count is fixed before evaluation; validation/test labels are never used.
validateattributes(trainData, {'numeric'}, {'2d','nonempty','finite'});
X = trainData(:,1:end-1);
y = trainData(:,end);
if isempty(X) || size(X,1) < 2
    error('Research:InvalidFeatures', 'Need at least two rows and one predictor.');
end
method = 'none';
if isfield(params,'NCA') && params.NCA
    method = 'nca';
end
if isfield(params,'featureMethod')
    method = validatestring(params.featureMethod, {'none','pca','nca'});
end
state.method = method;
state.inputFeatures = size(X,2);
state.mu = zeros(1,size(X,2));
state.sigma = ones(1,size(X,2));
state.indices = 1:size(X,2);
state.coeff = [];
state.center = [];
state.weights = [];
if strcmp(method,'none')
    return
end
state.mu = mean(X,1);
state.sigma = std(X,0,1);
state.sigma(state.sigma == 0) = 1;
Z = (X-state.mu)./state.sigma;
if strcmp(method,'pca')
    count = min([6, size(X,2), size(X,1)-1]);
    if isfield(params,'featureCount'), count = params.featureCount; end
    validateattributes(count, {'numeric'}, {'scalar','integer','positive','<=',min(size(X,2),size(X,1)-1)});
    [coeff,~,~,~,~,center] = pca(Z, 'NumComponents',count);
    state.coeff = coeff;
    state.center = center;
else
    count = max(1,ceil(size(X,2)/2));
    if isfield(params,'featureCount'), count = params.featureCount; end
    validateattributes(count, {'numeric'}, {'scalar','integer','positive','<=',size(X,2)});
    if numel(unique(y)) < 2
        error('Research:InvalidLabels', 'NCA requires at least two training classes.');
    end
    model = fscnca(Z,y,'Standardize',false,'Lambda',1/size(X,1));
    state.weights = model.FeatureWeights;
    if any(~isfinite(state.weights))
        error('Research:InvalidFeatures', 'NCA returned nonfinite weights.');
    end
    [~,ranking] = sort(state.weights,'descend');
    state.indices = sort(ranking(1:count));
end
end
