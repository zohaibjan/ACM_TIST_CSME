function tests = testOptimizers
tests = functiontests(localfunctions);
end
function setupOnce(testCase)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'tests','fixtures'));
rng(41,'twister');
end
function [models, data] = fixture()
truth = [0;10;0;10;0;10];
values = [truth, [0;0;0;10;10;10], [10;10;0;0;0;10]];
models = cell(1,3);
for k = 1:3
    models{k} = struct('name','DT','model',FixedClassifier(values(:,k)));
end
data = [(1:6).',truth];
end
function testBinaryVariants(testCase)
for variant = 1:8
    cost = @(bits,~,~) sum(bits);
    [mask, score, curve] = BPSO(8,6,variant,cost,5,{},{});
    verifySize(testCase,mask,[1,5]);
    verifyTrue(testCase,all(mask==0 | mask==1));
    verifyEqual(testCase,score,sum(mask));
    verifyTrue(testCase,all(isfinite(curve)) && all(diff(curve)<=0));
end
end
function testBinaryInvalidVariant(testCase)
threw = false;
try
    BPSO(8,6,9,@sum,5,{},{});
catch
    threw = true;
end
verifyTrue(testCase,threw);
end
function testGeneticScoreMatchesReturnedMask(testCase)
[models,data] = fixture();
opts = optimoptions('ga','PopulationSize',12,'MaxGenerations',6,'Display','off');
obj = runGA(models,data,opts);
verifyGreaterThan(testCase,sum(obj.chromosome),0);
verifyEqual(testCase,obj.fval,MyCost(obj.chromosome,models,data),'AbsTol',1e-12);
end
function testParetoScoresStayAligned(testCase)
[models,data] = fixture();
opts = optimoptions('gamultiobj','PopulationType','bitstring','PopulationSize',16, ...
    'MaxGenerations',6,'Display','off');
obj = runMultGA(models,data,opts);
verifySize(testCase,obj.chromosome,[1,3]);
verifyGreaterThan(testCase,sum(obj.chromosome),0);
verifyEqual(testCase,obj.fval(2),MyCost(obj.chromosome,models,data),'AbsTol',1e-12);
for k = 1:size(obj.paretoChromosomes,1)
    selected = psoPredict(models,data);
    wrong = selected(:,logical(obj.paretoChromosomes(k,:))) ~= data(:,end);
    if size(wrong,2)==1
        expected = mean(wrong);
    else
        pairs = nchoosek(1:size(wrong,2),2);
        expected = mean(mean(wrong(:,pairs(:,1)) & wrong(:,pairs(:,2)),1));
    end
    verifyEqual(testCase,obj.paretoScores(k,1),expected,'AbsTol',1e-12);
end
end
function testEmptyBinaryCostIsFinite(testCase)
[models,data] = fixture();
verifyEqual(testCase,MyCost([0,0,0],models,data),0);
end
function testFeatureSwarmFitsAndScores(testCase)
rng(19,'twister');
X = randn(32,3); y = 10*double(X(:,1)>0);
training = [X(1:24,:),y(1:24)];
validation = [X(25:end,:),y(25:end)];
params.classifiers = {'DT'};
opts = optimoptions('particleswarm','SwarmSize',4,'MaxIterations',2, ...
    'MaxStallIterations',2,'Display','off','UseParallel',false);
obj = optimizedFeatures(training,validation,params,opts);
mask = logical(obj.chromosome);
verifyTrue(testCase,any(mask));
models = trainFeatures(training(:,find(mask)),training(:,end),params);
[~, accuracy] = featureFusion(models,validation(:,find(mask)),validation(:,end));
verifyEqual(testCase,obj.fval,1-accuracy,'AbsTol',1e-12);
end
