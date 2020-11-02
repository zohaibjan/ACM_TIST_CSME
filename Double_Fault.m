function D=Double_Fault(C1,C2)
%Double fault used Pairwise classifiers
% C1 is the labels of thhe first class (Column)
% C2 is the labels of thhe second class(Column)
% T is the true labels (Column)

d=0;
for i=1:size(C1,1)
    if(C1(i,1)==0 && C2(i,1)==0)
        d=d+1;
    end
    
end

D=(d/size(C1,1));
