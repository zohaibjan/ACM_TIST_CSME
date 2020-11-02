function [fusion ,fusionAccuracy] = featureFusion(classifiers, X, y) 
   tempPredict = {};
   index = 1;
    for i=1:length(classifiers)
        try
            if strcmp(classifiers{1,i}.name, 'ANN') == 0
                tempPredict{index} = predict(classifiers{1,i}.model, X);
                index = index + 1;
            elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
                tempPredict{index} = getNNPredict(classifiers{1,i}.model, X);
                index = index + 1;
            end
        catch ME
            disp('In FEATURE FUSION');
            continue
        end
    end
    %% WHEN ALL IS DONE SAFELY    
    decisionMatrix = ones(length(X(:,1)), length(tempPredict));
    for j = 1:length(tempPredict)
        decisionMatrix(:,j) = cell2mat(tempPredict(j));
    end
    fusion = mode(decisionMatrix,2);
    fusionAccuracy = mean(y == fusion);
           
end

