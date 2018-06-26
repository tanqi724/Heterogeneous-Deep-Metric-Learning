function [F,D] = mmlkrLoss_global_K(L,X,Y,Ks,LFinal,xmiss,index_K)
%function [F,D] = mlkrLoss(L,X,Y,outdim,fun)
% Computes the mlkrLoss and gradient
% Input: 
% L : (dxd)x1 vector of the vectorized transformation matrix
% X : dxn input data
% Y : 1xn input labels
% 
% Optional:
%
% outdim : (default=d) , output dimensionality of L, in case of a rectangular matrix
% fun : a function of matrix L which is executed each iteration (e.g. to compute the validation error)
% i : the i_th features
%
% copyright Jake Gardner and Kilian Q. Weinberger, 2012
%

    [d,~,np] = size(X{1});
    [N,T] = size(Y);
    p = ceil(T/(np-1));
    F = 0;
    D_Final = zeros(N*N,1);
     
    for tt = 2 : T
        %%
        y = Y(:,tt)'; 
        %% Compute K from A, X
        tPeriod = ceil((tt+1)/p);
        for i = 1 : length(X)
            x = X{i};
            x = x(:,:,tPeriod);
            Lx = LFinal{i} * x;
            Kij = cmpKij(Lx);
            
            Ktemp = Ks{i};
            Ktemp(xmiss{i},xmiss{i}) = Kij;
            Ks{i} = Ktemp;
        end
        
        %% Pact the Kij{i} from 
        Kij = reshape(L, N, N);
        
        Ktemp = Ks{index_K};
        Ktemp(xmiss{index_K}==0,xmiss{index_K}==0) = Kij(xmiss{index_K}==0,xmiss{index_K}==0);
        Ks{index_K} = Ktemp;
        

        %% Compute K^h      
        K = Ks{1};
        for i = 2 : length(Ks)
            K = K + Ks{i};
        end
        
        % Calculate the Score function
        yhat = cmpYhat(K,y);

        F = F + sum((y-yhat').^2);
        if nargout > 1
           % Compute gradient
           D = mlkrGrad(x,y,yhat',Kij,K,xmiss{index_K});
           %D = vec(2*Ln*S);

           %% Constraint 
           D = reshape(D,N,N);
           D(xmiss{index_K},xmiss{index_K}) = 0;
           D = reshape(D,N*N,1);
           
           D_Final =  D_Final + D;
        end
    end
%    	temp = 0;
%     for i = 1 : length(Ks)
%        temp = temp + sum(sum( Ks{i}));
%     end
%     temp
    D = - D_Final; % Check that whether there is negative.
    D = D + reshape(2*min(Kij,0),N*N,1);
end


function O = mlkrGrad(X,Y,yhat,Kij,K,xmiss) 
    N = size(Y,2);
    dy=bsxfun(@minus,repmat(yhat',1,N),Y);
    den = 1./(sum(K,2) - diag(K)); %<- here (maybe)the Kij should be calculated as the sum_p K_p 
    den(den==Inf)=1/eps;
    dd=(yhat-Y)';
    O = (bsxfun(@times,bsxfun(@times,dy,dd),den)+bsxfun(@times,bsxfun(@times,dy',dd'),den')); %This is the derivate in terms of Kij
     
    %O = bsxfun(@times,dd,(bsxfun(@times,repmat(yhat,N,1),den') -  bsxfun(@times,repmat(yhat',1,N),den)));
end


function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end
