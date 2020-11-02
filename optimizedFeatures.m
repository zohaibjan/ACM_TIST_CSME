function obj=optimizedFeatures(trainData, validationData, params)
    warning('off','all');
    
    trainX = trainData(:,1:end-1);
    trainY = trainData(:, end);
    
    valX = validationData(:,1:end-1);
    valY = validationData(:, end);
    
    %set optimization function to PSOAF
    fun = @PSOAF;

    %lower and upper bounds. FOR BINARY
    lb=zeros(1,size(trainX,2));
    ub=ones(1,size(trainX,2));
    
    options = optimoptions('particleswarm','SwarmSize',min(100,size(trainX,2)));    
  
    %iterations.
    options.MaxIter=100;
    options.StallIterLimit=5;
    
    [best,fval,exitflag,output]=particleswarm(fun,size(trainX,2),lb,ub, options);

    %set returned objects.
    obj.chromosome=round(best); %best features
    obj.fval=fval;
    obj.output=output;
    
    %% OBJECTIVE FUNCTION
    function error=PSOAF(features)
        columns = round(features);
        columns = find(columns);
        classifierIndex = 1;
        classifiers = {};
        all = trainFeatures(trainX(:,columns), trainY, params);
        for temp = 1:length(all)
            classifiers{classifierIndex} = all{1,temp};
            classifierIndex = classifierIndex + 1;
        end
        [~, accuracy] = featureFusion(classifiers, valX(:,columns), valY);
        error = 1-accuracy;
    end
end
