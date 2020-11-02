Problem = {'breast-cancer-wisconsin', 'pima_diabetec', 'ecoli', 'haberman',...
            'ionosphere', 'liver', 'page-blocks', 'segment', 'sonar', 'statimag',...
            'thyroid', 'teaching', 'vowel', 'vehicle', 'wdbc', 'wine', 'banknote',...
            'balance', 'australian', 'iris', 'adult'
};  %FOR TNNLS


%% Model SETTINGS 
params.usePCA = false;
params.usePSOFeatures = false;
params.threshold = 0.90;                 % CLUSTER diss similarity ratio %55 works good
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS 
params.runPSO = true;
params.runMULTIGA = false;
params.usePSOClustering = false;         % THIS WILL OVERRIDE ALL OTHER FORMS OF CLUSTERING ALSO UB is root of 3
params.runGA = false;
params.noOfClusters = 3;                 % For nth root of clustering
params.useNClusters = false;
params.staticCluster = 4;                % ONLY USE WHEN useNClusters == true
params.useEnsembleSelection = false;      % ONLY USE ENSEMBLE THAT HAVE HIGHER ACCURACY HIGHER THAN 50% ON VALIDATION DATA
params.classifiers = {'ANN', 'SVM', 'KNN', 'DT', 'DISCR', 'NB'};
% params.classifiers = {'ADABOOST'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};
numRun = 3;

warning('off','all');
for p = 1:length(Problem)
    p_name = Problem{p};
    %% Create CROSS VALIDATION FOLDS
    numOfFolds = params.numOfFolds;
    data = load([pwd,filesep,'P-Data',filesep, p_name]);
    data = [normalize(data.X, 3) , data.y];
    data = rmmissing(data);
    cvFolds = cvpartition(data(:,end), 'KFold', numOfFolds);

    %% RECORD KEEPING VARIABLES 
    diversityWithoutClustering = [];
    diversityWithClustering = [];

    %% ITERATE OVER THE NUMBER OF FOLDS
    for fold=1:numOfFolds
        classifierIndex = 1;
        classifiers = {};        
    
        trainData = data(cvFolds.training(fold),:);
        testData = data(cvFolds.test(fold),:);
        
        %% SEPARATE VALIDATION DATA PER FOLD
        cv_vali_folds = cvpartition(trainData(:,end), 'holdout', 0.1);
        validationDataIndex = test(cv_vali_folds); 
        validationData = trainData(validationDataIndex,:);
        trainingSet =  ~validationDataIndex;
        trainData = trainData(trainingSet, :);
        
        noOfClusters = round(nthroot(size(trainData,1),params.noOfClusters));
               
        allClusters = zeros(length(trainData),1);
        totalClusters = 1;
        classifiersWithoutClustering = trainClassifiers(trainData, params); 
        %% GENERATE ALL CLUSTERS
        for clusters=1:noOfClusters
            %% CLUSTERINGS
            clusterIds = kmeans(trainData(:,1:end-1), clusters,'MaxIter',24000);  
                                                                        % 'MaxIter',500, 'Replicates',5,  'dist','sqeuclidean'
            for j=1:clusters
                allClusters(:,totalClusters) = (clusterIds == j); % indexes of clusters
                totalClusters = totalClusters + 1;
            end
        end
        
        generatedClusters(fold) = totalClusters;
        rClusters = zeros(length(trainData),1);
        n = 1;
        %% PRUNE CLUSTERS
        for i = 1:size(allClusters,2)
            if i == 1
                rClusters(:,1) = allClusters(:,1);
                continue;
            end
            
            for j=1:size(rClusters,2)
                similarity = length(intersect(find(rClusters(:,j)),find(allClusters(:,i)))) /...
                             length(union(find(rClusters(:,j)),find(allClusters(:,i))));
                if similarity > params.threshold
                    break;
                end
                if j == size(rClusters,2) % HAVE COMPARED WITH ALL THE RECORDED CLUSTERS
                    n = n + 1;
                    rClusters(:,n) = allClusters(:,i);
                end
            end
        end
        
        prunedClusters(fold) = size(rClusters,2);
        optIndex = 1;
        optimized = {};
        
        %% TRAIN ON CLUSTERS
        for j = rClusters
            trainCluster = trainData(find(j),:);
            noOfRecords = size(trainCluster,1);
            noOfClasses = length(unique(trainCluster(:,end)));

            if  noOfClasses > 1 && noOfRecords >= 10 %https://arxiv.org/abs/1211.1323
                    % TRY WITH 10/25/58/75 test samples
                    all = trainClassifiers(trainCluster, params);   
                    for temp = 1:length(all)
                        classifiers{classifierIndex} = all{1,temp};
                        classifierIndex = classifierIndex + 1;
                    end
                    if params.useEnsembleSelection == true
                        best = ensembleSelection(classifiers, validationData);
                        tIndex = 1;
                        for temp =length(optimized)+1 : length(optimized)+length(best) 
                            optimized(1,temp) = best(tIndex);
                            tIndex = tIndex + 1;
                        end
                    else
                        optimized = classifiers;
                    end
                    
            end
        end
        allPredictions = psoPredict(optimized, validationData);
        diversityWithClustering(fold) = Disagreement(allPredictions, validationData);
        
        all_Predictions = psoPredict(classifiersWithoutClustering, validationData);
        diversityWithoutClustering(fold) = Disagreement(all_Predictions, validationData);
        
     
    end 
       %% SAVE RESULTS
        if (exist([pwd filesep 'results-diversity.csv'], 'file') == 0)
            fid = fopen([pwd filesep 'results-diversity.csv'], 'w');
            fprintf(fid, '%s,%s, %s\n', ...
            'Data Set', 'Disagreement without Clustering', 'Disagreement with Clustering');
            fprintf(fid, '%s, %f, %f\n', p_name, mean(diversityWithoutClustering), mean(diversityWithClustering));
        elseif (exist([pwd filesep 'results-diversity.csv'], 'file') == 2)
            fid = fopen([pwd filesep 'results-diversity.csv'], 'a');
            fprintf(fid, '%s, %f, %f\n', p_name, mean(diversityWithoutClustering), mean(diversityWithClustering));
        end
end

