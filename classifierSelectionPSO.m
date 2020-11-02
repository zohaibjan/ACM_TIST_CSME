function obj=classifierSelectionPSO(classifierList, testData, p_name)
warning('off','all');
try
    warning('off','all');

        allPredictions = psoPredict(classifierList, testData);

        %set optimization function to PSOAF
        fun = @PSOAF;
        options2 = optimoptions('particleswarm','SwarmSize',100);

        options2.MaxIter=200;
        options2.StallIterLimit=20;

        lb=zeros(1,length(classifierList));
        ub=ones(1,length(classifierList));
        [best,fval,exitflag,output]=particleswarm(fun, length(classifierList),lb,ub,options2);
        obj.chromosome=round(best);
        obj.fval=fval;
        obj.output=output;
catch excp
    disp(sprintf('problem with %s', p_name));
end
    
    
        %% OBJECTIVE FUNCTION
        function error=PSOAF(c)
            c = round(c);
%             c = c > mean(c); % all those chromosomes which have personal best of higher than 0.5 
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