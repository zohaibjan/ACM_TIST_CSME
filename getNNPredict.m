function predict = getNNPredict(net,data)
    x = data';
    y = net(x);
    predict = vec2ind(y)';
end