%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  BPSO and VPSO source codes version 1.0                           %
%                                                                   %
%  Developed in MATLAB R2011b(7.13)                                 %
%                                                                   %
%  Author and programmer: Seyedali Mirjalili                        %
%                                                                   %
%         e-Mail: ali.mirjalili@gmail.com                           %
%                 seyedali.mirjalili@griffithuni.edu.au             %
%                                                                   %
%       Homepage: http://www.alimirjalili.com                       %
%                                                                   %
%   Main paper: S. Mirjalili and A. Lewis, "S-shaped versus         %
%               V-shaped transfer functions for binary Particle     %
%               Swarm Optimization," Swarm and Evolutionary         %
%               Computation, vol. 9, pp. 1-14, 2013.                %
%                                                                   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [o] = MyCost(c, classifierList, testData)
    
    allPredictions = psoPredict(classifierList, testData);
    c = find(c);
    decisionMatrix = ones(length(testData(:,end)), length(c));
    for i=1:length(c)            
        decisionMatrix(:,i) = allPredictions(:, c(i)) ;
    end
    decisionMatrix = mode(decisionMatrix, 2);
    error = mean(decisionMatrix ~= testData(:,end));
    
    o=error; % Modify or replace here according to your cost funciton
end

