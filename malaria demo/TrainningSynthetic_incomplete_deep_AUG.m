function [ Result ] = TrainningSynthetic_incomplete_deep_AUG( Y,Data )
disp('Training HDML-UF')
%% Data Loading
[N,TotalLength] = size(Y);
T = Data.T;
NPeriod = Data.NPeriod;
TotalT = size(Y,2);

X = Data.X; % cell array {X1, X2}
XMiss = Data.XMiss; % Cell array {Indicator for the presence of {X1,X2} } N x 1 in {0,1}: 1 for present, 0 for absent.
R = Data.RForTrain;
[~,~,np] = size(X{1});
P = ceil(T/(np-1));

%% Param
Hidden_size = 10;
AUG_Hidden_size = 5;

%% Init
AUG_Hidden = rand(AUG_Hidden_size,N);
LFinal = eye(Hidden_size+AUG_Hidden_size);

W = {};
for i = 1 : length(X)
    W{i} = 0.02*randn(Hidden_size,size(X{i},1));
end

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
SS = Data.ss(:,1:TotalT);
MissingWeighting = ones(1,TotalT);


while OutIter < 5
    iter = 1;
    while iter < 6    
        
        %%% Updtae Beta
        Beta = zeros(N+size(Data.TemporalX_ForTrain,1),N);
        recI = (Y-[zeros(N,1),residule]);
        for ii = 1 : N
            ss = SS(ii,:);
            ss = ss(1:end-1);
   
            recY = recI(ii,2:TotalT).*MissingWeighting(2:TotalT);
            recX = [ 
                Y(:,1:TotalT-1);
                Data.TemporalX_ForTrain(:,2:end)
            ];
            SX = recX;
            Beta(:,ii) = lasso(SX',recY','lambda',10);
 
            EMean(ii,2:TotalT) = Beta(:,ii)'*recX;
        end

        residule = Y - EMean;

        %% Update A
 

        x = zeros(Hidden_size+AUG_Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        
        % Data format: X{2} di x ni x np
        %   DMiss{2}: ni x 1. indicators of the locations in each...
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = [sigmoid(W{i}*X{i}(:,:,p)); AUG_Hidden(:,XMiss{i}) ];
            end
        end
        
        %[LFinal, hist] = minimize(LFinal(:),'mmlkrLoss_global',pars.maxiter,x(:,:,1:end-1),residule,LFinal,lambda);
        %LFinal = reshape(LFinal,Hidden_size+AUG_Hidden_size,Hidden_size+AUG_Hidden_size);
        
        
        %% Update W
        
        for i = 1 : length(W)
            [temp,hist] = minimize(W{i}(:),'embedding_loss_incomplete_AUG',pars.maxiter,X,residule,LFinal,W,i,XMiss,AUG_Hidden);
            W{i} = reshape(temp, Hidden_size, size(X{i},1));
        end
        
        
        [temp,hist] = minimize(AUG_Hidden(:),'hidden_state_loss_incomplete_AUG',pars.maxiter,X,residule,LFinal,W,XMiss);
        AUG_Hidden =  reshape(temp, AUG_Hidden_size, N ); 
        
        %% Update Residuel
        newResidule = residule;
        
        x = zeros(Hidden_size+AUG_Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = [sigmoid(W{i}*X{i}(:,:,p)); AUG_Hidden(:,XMiss{i}) ];
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


        iter = iter + 1;
     end
        %% Imputation
        
        [~,~,np] = size(X{1}); np = np -1;
        P = ceil(T/(np-1));
    
        residule = newResidule(:,2:end);
        x = zeros(Hidden_size+AUG_Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
        for i = 1 : length(X)      
            for p = 1: size(X{i},3)
                x(:,XMiss{i},p) = [sigmoid(W{i}*X{i}(:,:,p)); AUG_Hidden(:,XMiss{i}) ];
            end
        end

        
        for tt = 2 : 1 : T
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

        % Converge
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
        

        OutIter = OutIter + 1;      
end


Result.Beta = Beta; 
Result.LFinal = LFinal;  
Result.convergence = outconvergence; 
Result.P = p;
Result.W = W;
Result.AUG_Hidden = AUG_Hidden;
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