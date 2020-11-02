function program = mainProgram()
clear all;
clc;

Problem = dataSetNames();                 % Get list of dataset names

% Problem = {'hepatitis'};

%% Model SETTINGS
params.NCA = false;
params.numOfFolds = 10;                   % Create CROSS VALIDATION FOLDS
params.runPSO = false;                    % for optimization
params.noOfClusters = 5;                  % For nth root of clustering
params.diversity = {1,2,3,4,5,6};         % 1 for MD, 2 for DoublFault, 3 for Disagreement,
                                          % 4 for Q_Test, 5 for Interrater K, and 6 for Correlation
params.elimination = 3;                   % chances each classifier will be given 
params.classifiers = {'ANN', 'SVM', 'KNN', 'DT', 'DISCR', 'NB'};
% params.classifiers = {'ADABOOST'};
params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
params.trainFunctionDiscriminant = {'pseudoLinear','pseudoQuadratic'};
params.kernelFunctionSVM={'gaussian','polynomial','linear'};

%% MAIN LOOP
parfor i=1:length(Problem)
    for j =1:1
        p_name = Problem{i};
        disp(p_name);
        results = runTraining(p_name, params);
        saveResults(results, p_name);
    end
end
end





