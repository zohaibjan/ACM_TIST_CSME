function network = trainRBFNN(X, y, params , p)
   warning off;
    x = X;
    t = prepareTarget(y)';
    eg = 0.1; % sum-squared error goal
    sc = 4;    % spread constant
    net = newrb(x,t,eg,sc);
    network = net;
end
