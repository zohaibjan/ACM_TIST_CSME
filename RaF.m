% Problem = {'breast-cancer-wisconsin', 'pima_diabetec', 'ecoli', 'haberman',...
%                 'ionosphere', 'liver', 'page-blocks', 'segment', 'sonar', 'statimag',...
%                 'thyroid', 'teaching', 'vowel', 'vehicle', 'wdbc', 'wine', 'banknote',...
%                 'balance', 'australian', 'iris', 'adult'
%  };  %FOR TNNLS

useEnsemble = true;

Problem = {'adult'};
if useEnsemble == true
    learners = {'lpboost'};
else
    learners = {'fitcknn', 'fitcecoc', 'fitcnb', 'fitcdiscr', 'fitctree', 'trainNN'};
    % learners = {'fitcdiscr'};
end
for m=1:length(learners)
    for i=1:length(Problem)
        warning('off','all');
        p_name = Problem{i};
        %% Create CROSS VALIDATION FOLDS
        numOfFolds = 10;
        data = load([pwd,filesep,'P-Data',filesep, p_name]);
        data = [normalize(data.X, 3) , data.y];
        data = rmmissing(data);
        cvFolds = cvpartition(data(:,end), 'KFold', numOfFolds);
        
        %% RECORD KEEPING VARIABLES
        avgAccuracy = [];
        
        %% ITERATE OVER THE NUMBER OF FOLDS
        for fold=1:numOfFolds
            trainData = data(cvFolds.training(fold),:);
            testData = data(cvFolds.test(fold),:);
            
            trainX = trainData(:,1:end-1);
            trainY = trainData(:,end);
            testX = testData(:,1:end-1);
            testY = testData(:, end);
            
            if useEnsemble == true
                classifier = fitcensemble(trainX, trainY, 'Method',learners{m});
                predictY = predict(classifier, testX);
            else
                params.trainFunctionANN={'trainlm','trainbfg','trainrp','trainscg','traincgb','traincgf','traincgp','trainoss','traingdx'};
                params.kernelFunctionSVM={'gaussian','polynomial','linear'};
                maxTreeSize = length(unique(trainY));
                def.range=[10 30;                      % hidden neuron NN
                    1 9;                        % training function NN
                    100 5000;                   % NN epochs
                    ];
                p=floor(def.range(:,1)+(def.range(:,2)-def.range(:,1))*rand);
                learner = learners{m};
                func = str2func(learners{m});
                switch learner
                    case 'fitcknn'
                        classifier = func(trainX, trainY);
                    case 'fitcecoc'
                        classifier = func(trainX, trainY);
                    case 'fitcnb'
                        classifier = func(trainX, trainY, 'distribution', 'kernel');
                    case 'fitcdiscr'
                        classifier = func(trainX, trainY, 'discrimtype','pseudoLinear');
                    case 'fitctree'
                        classifier = func(trainX, trainY);
                    case 'trainNN'
                        classifier = func(trainX, trainY, params, p);
                end
                switch learner
                    case 'trainNN'
                        predictY = getNNPredict(classifier, testX);
                    otherwise
                        predictY = predict(classifier, testX);
                end
            end
            eval = Evaluate(testY, predictY);
            avgAccuracy(fold) = eval(1);
        end
        if (exist([pwd filesep 'results-RaF.csv'], 'file') == 0)
            fid = fopen([pwd filesep 'results-Raf.csv'], 'w');
            fprintf(fid, '%s,%s,%s\n', ...
                'Data Set', 'Method','Accuracy' ...
                );
        elseif (exist([pwd filesep 'results-Raf.csv'], 'file') == 2)
            fid = fopen([pwd filesep 'results-Raf.csv'], 'a');
        end
        fprintf(fid, '%s,', p_name);
        fprintf(fid, '%s,', learners{m});
        fprintf(fid, '%f±%f\n',mean(avgAccuracy),std(avgAccuracy));
        fclose(fid);
    end
end

