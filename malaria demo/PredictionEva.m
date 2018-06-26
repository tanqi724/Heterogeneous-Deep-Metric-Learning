function [ EVA ] = PredictionEva(TrueI,R,PredI)
%PREDICTIONEVA Summary of this function goes here
%   Detailed explanation goes here
TotalT = size(PredI,2);
Error = abs(TrueI - PredI).*(1-R);
Error(isnan(Error)) = max(max(Error));
mI = mean(TrueI,2);
%% r 
rMAE = nnsum(Error./kron(ones(1,TotalT),mI))/nnsum(1-R);
EVA.rMAE = rMAE;
rMSE = nssum(Error./kron(ones(1,TotalT),mI))/nnsum(1-R);
EVA.rMSE = rMSE;
rME = max(max(Error./kron(ones(1,TotalT),mI)));
EVA.rME = rME;

%%
MAE = nnsum(Error)/nnsum(1-R);
EVA.MAE = MAE;
MSE = nssum(Error)/nnsum(1-R);
EVA.MSE = MSE;
ME = max(max(Error));
EVA.ME = ME;

end

function s = nnsum(A)
    s = sum(sum(A));
end

function s = nssum(A)
    s = sum(sum(A.^2));
end