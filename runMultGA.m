function obj = runMultGA(classifierList, validationData, solverOptions)
%RUNMULTGA Pareto search for mean double fault and validation voting error.
% A singleton's double fault is its error rate. Empty masks retain model one.
% Select the smallest Pareto ensemble, then lowest error/double fault/index.
predictions = psoPredict(classifierList, validationData);
n = size(predictions,2);
if nargin < 3 || isempty(solverOptions)
    solverOptions = optimoptions('gamultiobj', 'PopulationType', 'bitstring', 'Display', 'off');
end
[best, scores, exitflag, output] = gamultiobj(@objective, n, [], [], [], [], ...
    zeros(1,n), ones(1,n), solverOptions);
masks = false(size(best));
for k = 1:size(best,1)
    masks(k,:) = selectionMask(best(k,:),0.5);
end
[~, order] = sortrows([sum(masks,2), scores(:,2), scores(:,1), (1:size(best,1)).']);
chosen = order(1);
obj.chromosome = double(masks(chosen,:));
obj.fval = scores(chosen,:);
obj.paretoChromosomes = double(masks);
obj.paretoScores = scores;
obj.exitflag = exitflag;
obj.output = output;
    function value = objective(weights)
        mask = selectionMask(weights,0.5);
        selected = predictions(:,mask);
        wrong = selected ~= validationData(:,end);
        if size(selected,2) == 1
            doubleFault = mean(wrong);
        else
            pairs = nchoosek(1:size(selected,2),2);
            doubleFault = mean(mean(wrong(:,pairs(:,1)) & wrong(:,pairs(:,2)),1));
        end
        value = [doubleFault, mean(votePredictions(selected) ~= validationData(:,end))];
    end
end
