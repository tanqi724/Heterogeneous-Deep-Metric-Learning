function [EVA] = Testing_incomplete_deep_AUG(YTe,Result,Data)
%% HyperSetting
T = Data.T;
R = Data.RForTest;
YTe(R==0)=0;
[N,T]= size(YTe);
%% Extract Information
Beta = Result.Beta;
X = Data.X;
LF = Result.LFinal;
XMiss = Data.XMiss;
W = Result.W;
hidden_size = size(W{1},1);
AUG_Hidden = Result.AUG_Hidden;
AUG_Hidden_size = size(AUG_Hidden,1);

x = zeros(hidden_size+AUG_Hidden_size,N,size(X{1},3)); % hidden_size, # of locations, # of period
for i = 1 : length(X)      
    for p = 1: size(X{i},3)
        x(:,XMiss{i},p) = [sigmoid(W{i}*X{i}(:,:,p)); AUG_Hidden(:,XMiss{i}) ];
    end
end

Lx = LF * x(:,:,end);
Kij = cmpKij(Lx);
                
Kij=Kij-diag(diag(Kij)-diag(0));
TotalT = size(YTe,2);

%% Initlization

yhat = YTe;
yhat(Data.RForTest==0) = 0; 
v = zeros(T,1);

%% Return initial
EMean = zeros(N,T);
SS = Data.ssForTest;
SpatialVariance = zeros(N,T);
for tt = 2 : 1 : T
for ii = 1 : N
    ss = SS(ii,:);
    X = [ 
        yhat(:,tt-1);
        Data.TemporalX_ForTest(:,tt)
    ];
    EMean(ii,tt) = Beta(:,ii)'*X;
end
    MissingIdxList = find(R(:,tt)== 0);
    ObIdx =  find(R(:,tt)== 1);
    
    INDtemp = EMean(:,tt) < 0;
    EMean(INDtemp,tt) = 0;
    
    if length(ObIdx)>0*N
     
        ygap = YTe(:,tt) - EMean(:,tt);
        ygap(MissingIdxList) = 0; 
        for missingidx = 1:length(MissingIdxList) 
            SUM_KIJ = sum(Kij(MissingIdxList(missingidx),ObIdx));
            if SUM_KIJ == 0
                SUM_KIJ = eps;
            end
            v(tt) = Kij(MissingIdxList(missingidx),ObIdx)*(ygap(ObIdx))/SUM_KIJ;
            SpatialVariance(MissingIdxList(missingidx),tt) = v(tt);
            yhat(MissingIdxList(missingidx),tt) = EMean(MissingIdxList(missingidx),tt)+ v(tt);
        end
    else 
        yhat(MissingIdxList,tt) = EMean(MissingIdxList,tt);
    end
    INDtemp = yhat(:,tt) < 0;
    yhat(INDtemp,tt) = 0;
end


[ EVA ] = PredictionEva(Data.IForTest,R,yhat);
disp('MAE is')
disp(EVA.MAE)
EVA.yhat = yhat;
end

function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end