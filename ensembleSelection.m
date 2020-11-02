function ensemble = ensembleSelection(classifiers, validationData)

    X = validationData(:,1:end-1);
    y = validationData(:,end);
    
    accuracies = zeros(1, length(classifiers));
    for i=1:length(classifiers)
        try
            if strcmp(classifiers{1,i}.name, 'ANN') == 0
                 accuracies(1,i) = mean(predict(classifiers{1,i}.model, X) == y);
            elseif strcmp(classifiers{1,i}.name, 'ANN') == 1
                 accuracies(1,i) = mean(getNNPredict(classifiers{1,i}.model, X) == y);
            end
        catch ME
            disp('');
            continue
        end
    end
    
    if length(classifiers) == 1
        ensemble = classifiers;
        return
    end
    threshold = min(accuracies);
    selected = [];
    
    for i=1:length(accuracies)
        if accuracies(i) > threshold
            selected(i) = 1;
        else
            selected(i) = 0;
        end
    end
    selected = find(selected);
    ensemble = {};
    
    for j=1:length(selected)
        ensemble{1,j} = classifiers{1,selected(j)};
    end
end





