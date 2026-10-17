function acc = fusion(classifiers, selections, data)
%FUSION Score only explicitly selected classifiers.
if numel(selections) ~= numel(classifiers) || any(~ismember(selections(:), [0 1]))
    error('Research:InvalidSelection', 'Supply one binary flag per classifier.');
end
predictions = predictionMatrix(classifiers(logical(selections)), data(:, 1:end-1));
acc = mean(votePredictions(predictions) == numericLabels(data(:, end)));
end
