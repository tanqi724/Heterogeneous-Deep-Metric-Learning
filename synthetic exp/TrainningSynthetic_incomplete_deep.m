function [ Result ] = TrainningSynthetic_incomplete_deep( Y,Data )
%% Data Loading 
X = Data.X;
XMiss = Data.XMiss; 
R = Data.RForTrain;
P = Data.P;

disp('Training HDML')
%% Init
Hidden_size = 10;
LFinal = eye(Hidden_size);
W = {};
for i = 1 : length(X)
    W{i} = 0.02*randn(Hidden_size,size(X{i},1));
end

[N,TotalLength] = size(Y);
NPeriod = Data.NPeriod;
X = Data.X;
TotalT = size(Y,2);
Y = PreProcess(Y,R); 
Beta = zeros(N,N);
residule = zeros(N,TotalLength-1);

%%

pars.maxiter=-20;
pars.function=[];

convergence = [];
outconvergence = [];

EMean = Y;
OutIter = 1;
%%

LAG = 1;
while OutIter < 3
    iter = 1;
    while iter < 3    
        
        %%% Updtae Beta
        Beta = zeros(2*N+size(Data.TemporalX_ForTrain,1),N);
        recI = (Y-[zeros(N,1),residule]);
        
        recI = recI(:,LAG+1:end);
        
        recX = [];
        for lag_temp = 1 : LAG
            recX = [ recX;
                    Y(:,lag_temp:TotalT-LAG+lag_temp-1);
                    Data.TemporalX_ForTrain(:,lag_temp:TotalT-LAG+lag_temp-1);
                    Y(:,lag_temp:TotalT-LAG+lag_temp-1).*Data.TemporalX_ForTrain(:,lag_temp:TotalT-LAG+lag_temp-1);
                ];
        end
        for ii = 1 : N
            recY = recI(ii,:);
            SX = recX;
            Beta(:,ii) = lasso(SX',recY','lambda',10);
 
            EMean(ii,2:TotalT) = Beta(:,ii)'*recX;
        end

        residule = Y - EMean;

        %% Update A
 
        x = zeros(Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        
        % Data format: X{2} di x ni x np
        %   DMiss{2}: ni x 1. indicators of the locations in each...
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = sigmoid(W{i}*X{i}(:,:,p));
            end
        end
         
        
        %% Update W
        
        for i = 1 : length(W)
           [temp, hist] = minimize(W{i}(:),'embedding_loss_incomplete',pars.maxiter,X,residule,LFinal,W,i,XMiss);
           W{i} = reshape(temp, Hidden_size, size(X{i},1)); 
        end
        
        %% Update Residuel
        newResidule = residule;
        
        x = zeros(Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = sigmoid(W{i}*X{i}(:,:,p));
            end
        end

        for ii = 1 : 1 : 1
            for iPeriod = 1 : NPeriod-1
                        
                Lx = LFinal * x(:,:,iPeriod);
                Kij = cmpKij(Lx);
        
                for tt = 1 : P                  
                     newResidule(:,(iPeriod-1)*P+tt) = cmpYhat(Kij,residule(:,(iPeriod-1)*P+tt));
                end
            end
        end

        newResidule(:,1) = zeros(N,1);     
        sumResidule = abs(Y - EMean -newResidule);
        sumResidule(R == 0 ) = 0;
        sumResidule = sum(sum(sumResidule))/nsum(R == 1);
        residule = newResidule(:,2:end);

        if (iter > 1 && (abs(sumResidule -  convergence(end))/ convergence(end) < 1e-4 ))
            convergence = [convergence; sumResidule];
            break;
        else       
            convergence = [convergence; sumResidule];
        end
    
        iter = iter + 1
     end
        %% Imputation
        
        [~,~,np] = size(X{1});
        P = ceil(TotalT/(np-1));
    
        residule = newResidule(:,2:end);
        x = zeros(Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = sigmoid(W{i}*X{i}(:,:,p));
            end
        end
        
        for tt = 2 : 1 : TotalT
            MissingIdxList = find(R(:,tt)== 0);
            ObIdx =  find(R(:,tt)== 1);
            ygap = Y(:,tt) - EMean(:,tt);


            iPeriod = ceil((tt+1)/P);
            Lx = LFinal * x(:,:,iPeriod);
            Kij = cmpKij(Lx);

            for missingidx = 1:length(MissingIdxList) 
                temp = Kij(MissingIdxList(missingidx),ObIdx)*ygap(ObIdx)/sum(Kij(MissingIdxList(missingidx),ObIdx));
                if isnan(temp)
                    temp = 0;
                end
                Y(MissingIdxList(missingidx),tt) = EMean(MissingIdxList(missingidx),tt)+ temp;
            end
        end

        Y(isnan(Y)) = 1;
        residule(isnan(residule)) = 1;

        Episilon = abs(Y - EMean-[zeros(N,1),residule]);
        Episilon(R == 0) = 0;
        Episilon = nsum(Episilon);
        if (OutIter > 1 && (abs(outconvergence(end)  - Episilon )/Episilon < 1e-6 )) ||  Episilon ==0
                outconvergence = [outconvergence; Episilon];
                break;
            else
                outconvergence = [outconvergence; Episilon];
        end

        OutIter = OutIter + 1       
end


Result.Beta = Beta; 
Result.LFinal = LFinal;  
Result.convergence = outconvergence; 
Result.P = P;
Result.W = W;

end


function [Kij] = cmpKij(LX)
    dist = distance(LX);
    Kij = exp(-dist);
end

function Y = PreProcess(Y,R)
[MissingIdxRow,MissingIdxCol] = find(R==0);
for ii = 1 : length(MissingIdxRow)
    tt = MissingIdxCol(ii);
    ll = MissingIdxRow(ii);

    Y(ll,tt) = mean( Y(R(:,tt)==1,tt));
end
end


function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end
function [S] = nsum(A)
S = sum(sum(A));
end