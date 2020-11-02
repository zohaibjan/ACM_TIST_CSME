function [selections, div] = rBCS(classifiers, data, params)
div = 0;
selections = zeros(1,length(classifiers));
discarded = ones(1,length(classifiers))*params.elimination;
X = data(:, 1:end-1);
Y= data(:, end);
all_accuracies = [];
for c=1:length(classifiers)
    [all_accuracies(c), y_all] = accuracy(classifiers{1,c},X, Y);
end

selections(find(all_accuracies == max(all_accuracies))) = 1;
discarded(find(all_accuracies == max(all_accuracies))) = 0;


previousDiversity = 0;
k=find(selections);
numberOfComparisons = 0;
for m=1:length(k)-1
    for n=m+1:length(k)
        [model1, y1] = accuracy(classifiers{1, m}, X, Y);
        [model2, y2] = accuracy(classifiers{1, n}, X, Y);
        previousDiversity = previousDiversity + Double_Fault((y1==Y), (y2==Y));
        numberOfComparisons = numberOfComparisons + 1;
    end
end

averageDiversity = previousDiversity / numberOfComparisons;

while sum(discarded)>0
    [ensembleAcc_1, y1] = fusion(classifiers, selections, X, Y);
    next = randi([1 length(classifiers)], 1, 1);
    while any(find(selections) == next) && any(find(discarded == 0) == next)
        next = randi([1 length(classifiers)], 1, 1);
    end
    temp = selections;
    temp(next) = 1;
    [ensembleAcc_2, y2] = fusion(classifiers, temp, X, Y);  
    [aOfNewClassifier, yOfNewClassifier] = accuracy(classifiers{1, next}, X, Y); 
    newDiversity = Double_Fault((y1==Y), (yOfNewClassifier==Y));
    
    if ensembleAcc_1 >= ensembleAcc_2 
        discarded(next) = discarded(next) - 1;
    elseif ensembleAcc_2 > ensembleAcc_1 
        selections(next) = 1;
        discarded(next) = 0;
        averageDiversity = newDiversity;
     elseif ensembleAcc_2 == ensembleAcc_1 && newDiversity > averageDiversity
        selections(next) = 1;
        discarded(next) = 0;
        averageDiversity = newDiversity;
        div = div + 1;
    end
end
end

function [acc, y]=accuracy(classifier, X, Y)
prediction = [];
try
    if strcmp(classifier.name, 'SVM') == 1
        prediction = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'KNN') == 1
        prediction = predict(classifier.model, X);       
    elseif strcmp(classifier.name, 'DT') == 1
        prediction = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'NB') == 1
        prediction = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'DISCR') == 1
        prediction = predict(classifier.model, X);
    elseif strcmp(classifier.name, 'ANN') == 1
        prediction = getNNPredict(classifier.model, X);
    end
catch ME
    disp(sprintf('accuracy function %s',ME.identifier));
end
acc = mean(prediction == Y);
y = prediction;
end

function [acc, y]=fusion(classifiers, selections, X, Y)
c = find(selections);
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
acc = mean(decisionMatrix == Y);
y = decisionMatrix;
end