function acc=fusion(classifiers, selections, data)
c = find(selections);
X = data(:, 1:end-1);
Y = data(:, end);
decisionMatrix = ones(length(X(:,1)), length(c));
index = 1;
for i=c
    try
        if strcmp(classifiers{1,i}.name, 'SVM') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'KNN') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DT') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'NB') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'DISCR') == 1
            decisionMatrix(:,index) = predict(classifiers{1,i}.model, X);
        elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
            decisionMatrix(:,index) = getNNPredict(classifiers{1,i}.model, X);
        end
        index = index + 1;
    catch ME
        disp(sprintf('fusion causing errors: %s',ME.identifier));
    end
end
decisionMatrix = mode(decisionMatrix, 2);
% acc = mean(decisionMatrix == Y);
acc = Evaluate(Y, decisionMatrix);
end