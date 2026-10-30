function tests = testFeatureTransform
% Training/held-out boundaries and actual toolbox PCA/NCA execution.
tests = functiontests(localfunctions);
end
function setupOnce(testCase)
testCase.TestData.originalPath = path;
addpath(fileparts(fileparts(mfilename('fullpath'))), '-begin');
end
function teardownOnce(testCase)
path(testCase.TestData.originalPath);
end
function testFrozenPcaAndLabels(testCase)
rng(42); X = randn(40,5); X(:,5) = 1;
D = [X, repelem([0;10],20)];
p.featureMethod = 'pca'; p.featureCount = 2;
s = fitFeatureTransform(D,p);
held = [1000+randn(8,5), (1:8)'];
Z = applyFeatureTransform(held,s);
verifyEqual(testCase,s.mu,mean(X,1));
verifySize(testCase,Z,[8 3]);
verifyEqual(testCase,Z(:,end),held(:,end));
verifyTrue(testCase,all(isfinite(Z(:))));
verifyEqual(testCase,applyFeatureTransform(D,s), ...
    [((X-s.mu)./s.sigma-s.center)*s.coeff,D(:,end)],'AbsTol',1e-12);
end
function testActualNcaFixedFeatureCount(testCase)
rng(42); D = [randn(40,5), repelem([0;10],20)];
p.NCA = true; p.featureCount = 2;
s = fitFeatureTransform(D,p);
verifyEqual(testCase,numel(unique(s.indices)),2);
verifySize(testCase,applyFeatureTransform(D,s),[40 3]);
verifyEqual(testCase,applyFeatureTransform([D(:,1:5),99*ones(40,1)],s), ...
    [((D(:,s.indices)-s.mu(s.indices))./s.sigma(s.indices)),99*ones(40,1)],'AbsTol',1e-12);
end
function testNoneAndInvalidFeatures(testCase)
D = [1 2 0; 3 4 10; 5 6 0];
s = fitFeatureTransform(D,struct());
verifyEqual(testCase,applyFeatureTransform(D,s),D);
verifyError(testCase,@() applyFeatureTransform([1 0],s),'Research:InvalidFeatures');
end
