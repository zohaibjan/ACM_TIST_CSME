function [selections, addedDiv] = selectByValidation(predictions, truth, elimination)
%SELECTBYVALIDATION Bounded randomized strict-accuracy selection on cached votes.
truth = numericLabels(truth);
if size(predictions, 1) ~= numel(truth)
    error('Research:PredictionSize', 'Validation predictions and labels must align.');
end
votePredictions(predictions); % Validate the entire candidate pool first.
if ~isnumeric(elimination) || ~isscalar(elimination) || ~isfinite(elimination) || ...
        elimination < 0 || elimination ~= floor(elimination)
    error('Research:InvalidElimination', 'elimination must be a nonnegative integer.');
end
accuracies = mean(bsxfun(@eq, predictions, truth), 1);
selected = accuracies == max(accuracies);
remaining = repmat(elimination, 1, size(predictions, 2));
remaining(selected) = 0;
addedDiv = 0; % Historical diversity-tie branch was unreachable; retain strict improvement.
while true
    eligible = find(~selected & remaining > 0);
    if isempty(eligible)
        break;
    end
    next = eligible(randi(numel(eligible)));
    currentAccuracy = mean(votePredictions(predictions(:, selected)) == truth);
    candidate = selected;
    candidate(next) = true;
    candidateAccuracy = mean(votePredictions(predictions(:, candidate)) == truth);
    if candidateAccuracy > currentAccuracy
        selected = candidate;
        remaining(next) = 0;
    else
        remaining(next) = remaining(next) - 1;
    end
end
selections = double(selected);
end
