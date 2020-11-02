function D=Disagreement(C1,C2)
%Disagreement test used Pairwise classifiers
% C1 is the labels of thhe first class (Column)
% C2 is the labels of thhe second class(Column)
% T is the true labels (Column)

N=zeros(2,4);
a=0;b=0;c=0;d=0;
for i=1:size(C1,1)
    if(C1(i,1)==1 && C2(i,1)==1)
        a=a+1;
    end
    
    if(C1(i,1)==0 && C2(i,1)==0)
        d=d+1;
    end
    
    if(C1(i,1)==1 && C2(i,1)==0)
        b=b+1;
    end
    
    if(C1(i,1)==0 && C2(i,1)==1)
        c=c+1;
    end
end

D=(b+c)/size(C1,1);