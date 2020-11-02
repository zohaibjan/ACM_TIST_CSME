function obj = runGA(classifierList, testData)
    warning('off','all');
    allPredictions = psoPredict(classifierList, testData);
    
    %set optimization function to PSOAF
    fun = @GA;
        
    %lower and upper bounds. FOR BINARY
    lb=zeros(1,length(classifierList));
    ub=ones(1,length(classifierList));
    [best,fval,exitflag,output] = ga(fun, length(classifierList),[],[],[],[],lb,ub);

    %set returned objects.
    obj.chromosome=round(best);
    obj.fval=fval;
    obj.output=output;

    %% OBJECTIVE FUNCTION
    function error=GA(c)
%             c = round(m);
        c = c > mean(c); % all those chromosomes which have personal best of higher than 0.5 
        c = find(c);
        %% **************************** %%
        decisionMatrix = ones(length(testData(:,end)), length(c));
        for i=1:length(c)            
            decisionMatrix(:,i) = allPredictions(:, c(i)) ;
        end
        decisionMatrix = mode(decisionMatrix, 2);
        error = mean(decisionMatrix ~= testData(:,end));
    end
end

