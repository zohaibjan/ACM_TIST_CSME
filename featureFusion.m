function [fusion, fusionAccuracy] = featureFusion(classifiers, X, y)
%FEATUREFUSION Propagate prediction failures rather than adding fake votes.
fusion = votePredictions(predictionMatrix(classifiers, X));
fusionAccuracy = mean(fusion == numericLabels(y));
end
