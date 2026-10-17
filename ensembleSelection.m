function ensemble = ensembleSelection(classifiers, validationData)
X = validationData(:, 1:end-1);
y = validationData(:, end);
% Preserve the above-minimum rule; all-tied valid pools retain all models.
predictions = predictionMatrix(classifiers, X);
y = numericLabels(y);
if size(predictions, 1) ~= numel(y)
    error('Research:PredictionSize', 'Validation labels must align with samples.');
end
accuracies = mean(bsxfun(@eq, predictions, y), 1);
selected = accuracies > min(accuracies);
if ~any(selected)
    selected(:) = true;
end
ensemble = classifiers(selected);
end
