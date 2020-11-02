function obj=classifierSelectionPSO(trainData, validationData, params)
     warning('off','all');
   
    %set optimization function to PSOAF
    fun = @PSOAF;

    %lower and upper bounds. FOR BINARY
    lb=1;
%     ub=round(nthroot(length(trainData),3));
    ub=round(nthroot(size(trainData,1),params.noOfClusters));
    %define options.
    options = optimoptions('particleswarm','SwarmSize',100,...
        'ObjectiveLimit',0, 'HybridFcn',@fmincon);  % BECAUSE WE NEED CONSTRAINED HYBRID FUNCTION
  
    %iterations.
    options.MaxIter=10;
    options.StallIterLimit=2;
    
    [best,fval,exitflag,output]=particleswarm(fun,1,lb,ub, options);

    %set returned objects.
    obj.chromosome=round(best);
    obj.fval=fval;
    obj.output=output;

    
    %% OBJECTIVE FUNCTION
    function error=PSOAF(noOfClusters)
%         disp(sprintf('no of clusters chosen %i', round(noOfClusters)));
        
        %% CLUSTERINGS
        if noOfClusters > length(trainData)
           error = 100;
           return
        end
        clusterIds = kmeans(trainData(:,1:end-1), round(noOfClusters), 'dist','sqeuclidean');  
                                                                    % 'MaxIter', 500, 'Replicates',5,
        classifiers = {};
        classifierIndex = 1;
        for j = 1:noOfClusters
            recordIndexes  = (clusterIds == j);
            trainCluster = trainData(recordIndexes,:);
            noOfRecords = size(trainCluster,1);
            noOfClasses = length(unique(trainCluster(:,end)));

            if  noOfClasses > 1 && noOfRecords >= 10 %https://arxiv.org/abs/1211.1323
                    % TRY WITH 10/25/58/75 test samples
                    all = trainClassifiers(trainCluster, params);   
                    for temp = 1:length(all)
                        classifiers{classifierIndex} = all{1,temp};
                        classifierIndex = classifierIndex + 1;
                    end
            end
           [fusion1, error] = fusionPSO(classifiers, validationData);
        end
    end
end
