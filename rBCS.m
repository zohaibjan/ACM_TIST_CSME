function [selections, div] = rBCS(classifiers, data, params)
%RBCS Validation-only selection with bounded retries and cached predictions.
predictions = predictionMatrix(classifiers, data(:, 1:end-1));
[selections, div] = selectByValidation(predictions, data(:, end), params.elimination);
end
