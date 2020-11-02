function network = trainNN(X, y, params , p)
    x = X';
    t = prepareTarget(y)';

    % Choose a Training Function
    % For a list of all training functions type: help nntrain
    % 'trainlm' is usually fastest.
    % 'trainbr' takes longer but may be better for challenging problems.
    % 'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = params.trainFunctionANN{p(2)};  

    % Create a Pattern Recognition Network
    hiddenLayerSize = p(1);
    net = patternnet(hiddenLayerSize, trainFcn);
    net=configure(net,x,t);
    net.trainParam.showWindow=0;
    net.trainParam.show=50;
    net.trainParam.epochs= p(3);
    net.trainParam.goal=1.0e-3;
    
   
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 80/100;
    net.divideParam.valRatio = 20/100;
%     net.divideParam.testRatio = 10/100;

    % Train the Network
    [net,tr] = train(net,x,t);
    network = net;
end
