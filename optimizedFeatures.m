function obj = optimizedFeatures(trainData, validationData, params, solverOptions)
%OPTIMIZEDFEATURES Train on training rows; score a nonempty feature mask on validation.
trainX = trainData(:,1:end-1);
trainY = trainData(:,end);
valX = validationData(:,1:end-1);
valY = validationData(:,end);
n = size(trainX,2);
if n < 1 || size(valX,2) ~= n
    error('Research:InvalidFeatures', 'Training and validation feature counts must agree.');
end
if nargin < 4 || isempty(solverOptions)
    solverOptions = optimoptions('particleswarm', 'SwarmSize', min(100,max(2,n)), ...
        'MaxIterations',100, 'MaxStallIterations',5, 'Display','off');
end
[best, score, exitflag, output] = particleswarm(@objective,n,zeros(1,n),ones(1,n),solverOptions);
obj.chromosome = double(selectionMask(best,0.5));
obj.fval = score;
obj.exitflag = exitflag;
obj.output = output;
    function value = objective(weights)
        mask = selectionMask(weights,0.5);
        % Keep stochastic learner initialisation reproducible across evaluations.
        state = rng;
        restore = onCleanup(@() rng(state));
        rng(1,'twister');
        classifiers = trainFeatures(trainX(:,mask),trainY,params);
        [~, accuracy] = featureFusion(classifiers,valX(:,mask),valY);
        value = 1-accuracy;
    end
end
