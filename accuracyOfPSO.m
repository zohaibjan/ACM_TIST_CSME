function [accuracy, fMeasures, decisionMatrix, BCP]=accuracyOfPSO(classifiers, chromosome, testData)     
    c = find(chromosome);
    X = testData(:, 1:end-1);
    y = testData(:,end);
    BCP.SVM = 0;
    BCP.ANN = 0;
    BCP.DT = 0;
    BCP.KNN = 0;
    BCP.NB = 0;
    BCP.DISCR = 0;
    decisionMatrix = ones(length(testData(:,1)), length(c));
    for i=1:length(c)            
        try
            if strcmp(classifiers{1,i}.name, 'SVM') == 1
                decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
                BCP.SVM = BCP.SVM + 1;
            elseif strcmp(classifiers{1,i}.name, 'KNN') == 1
                decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
                BCP.KNN = BCP.KNN + 1;
            elseif strcmp(classifiers{1,i}.name, 'DT') == 1
                decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
                BCP.DT = BCP.DT + 1;
            elseif strcmp(classifiers{1,i}.name, 'NB') == 1
                decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
                BCP.NB = BCP.NB + 1;
            elseif strcmp(classifiers{1,i}.name, 'DISCR') == 1
                decisionMatrix(:,i) = predict(classifiers{1,i}.model, X);
                BCP.DISCR = BCP.DISCR + 1;
            elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
                decisionMatrix(:,i) = getNNPredict(classifiers{1,i}.model, X);
                BCP.ANN = BCP.ANN + 1;
            end
        catch ME
            disp(sprintf('IN ACCURACY OF PSO: %s',ME.identifier));
            continue
        end 
    end
    
    decisionMatrix = mode(decisionMatrix, 2);
    accuracy = mean(decisionMatrix == y);
    fMeasures = confusionmatStats(y, decisionMatrix);       
end