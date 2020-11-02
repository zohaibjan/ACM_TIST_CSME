%%Disaggreement test
function diversity = calcDiversity(classifiers, validationData)

    X = validationData(:,1:end-1);
    y = validationData(:,end);
    count = 1;
    similar = ones(1, length(classifiers));
    for i=1:length(classifiers)-1
        for j = i+1:length(classifiers)
           allDiv(count) =  mean (predict(classifiers{1,i}, X) ~= predict(classifiers{1,j},X));
           count = count + 1; 
        end
    end
    diversity = similar;
end





