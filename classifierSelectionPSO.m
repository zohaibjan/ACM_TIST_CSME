function obj=classifierSelectionPSO(classifierList, testData, p_name)

try

        allPredictions = psoPredict(classifierList, testData);

        %set optimization function to PSOAF
        fun = @PSOAF;
        options2 = optimoptions('particleswarm','SwarmSize',100);

        options2.MaxIterations=200;
        options2.MaxStallIterations=20;

        lb=zeros(1,length(classifierList));
        ub=ones(1,length(classifierList));
        [best,fval,exitflag,output]=particleswarm(fun, length(classifierList),lb,ub,options2);
        obj.chromosome=double(selectionMask(best, 0.5));
        obj.fval=fval;
        obj.output=output;
catch excp
    rethrow(excp);
end
    
    
        %% OBJECTIVE FUNCTION
        function error=PSOAF(c)
            c = selectionMask(c, 0.5);
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