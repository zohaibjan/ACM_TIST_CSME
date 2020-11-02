function obj = runMultGA(classifierList, testData)
    warning('off','all');
    allPredictions = psoPredict(classifierList, testData);
    
    %set optimization function to PSOAF
    fun = @MULTGA;
        
    %lower and upper bounds. FOR BINARY
    lb=zeros(1,length(classifierList));
    ub=ones(1,length(classifierList));
    
    options = optimoptions('gamultiobj','populationtype','bitstring');
%     options.populationtype = 'bitstring';
    
    [best,fval,exitflag,output] = gamultiobj(fun, length(classifierList),[],[],[],[],[],[], options);

    %set returned objects.
    obj.chromosome=round(best);
    obj.chromosome = obj.chromosome(find(sum(obj.chromosome,2) == min(sum(obj.chromosome,2))),:)
    obj.fval=fval;
    obj.output=output;

    %% OBJECTIVE FUNCTION
    function error=MULTGA(clsf)
        c = round(clsf);
        c = find(c);
        %% **************************** %%
        decisionMatrix = ones(length(testData(:,end)), length(c));
        for i=1:length(c)            
            decisionMatrix(:,i) = allPredictions(:, c(i)) ;
        end
        error(1)= Double_Fault(allPredictions,testData);
        decisionMatrix = mode(decisionMatrix, 2);
        error(2) = mean(decisionMatrix ~= testData(:,end));
    end
end

