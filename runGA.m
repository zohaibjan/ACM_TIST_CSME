function obj = runGA(classifierList, validationData, solverOptions)
%RUNGA Minimise validation error; use one nonempty mask policy throughout.
% Test data must remain held out from this selection step.
predictions = psoPredict(classifierList, validationData);
n = size(predictions, 2);
if nargin < 3 || isempty(solverOptions)
    solverOptions = optimoptions('ga', 'Display', 'off');
end
[best, score, exitflag, output] = ga(@objective, n, [], [], [], [], ...
    zeros(1,n), ones(1,n), [], solverOptions);
obj.chromosome = double(selectionMask(best, 0.5));
obj.fval = score;
obj.exitflag = exitflag;
obj.output = output;
    function value = objective(weights)
        mask = selectionMask(weights, 0.5);
        value = mean(votePredictions(predictions(:,mask)) ~= validationData(:,end));
    end
end
