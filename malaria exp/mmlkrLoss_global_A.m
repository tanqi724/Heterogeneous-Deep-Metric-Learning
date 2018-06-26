function [F,D] = mmlkrLoss_global_A(L,X,Y,Ks,xmiss,index_A,outdim,fun)
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

    [d,N,np] = size(X);
    [N,T] = size(Y);
    p = ceil(T/(np-1));
    if nargin<7,outdim=d;end;
    F = 0;
    D_Final = zeros(d*d,1);
    
   
        
    for tt = 2 : T
        % Compute New Kij
        tPeriod = ceil((tt+1)/p);
        x = X(:,:,tPeriod);
        %y = Y(xmiss,tt)';
        y = Y(:,tt)';
        Ln = reshape(L, outdim, d);
        Lx = Ln * x;
        Kij = cmpKij(Lx);
        
        Ktemp = Ks{index_A};
        Ktemp(xmiss,xmiss) = Kij;
        Ks{index_A} = Ktemp;
        
        K = Ks{1};
        for i = 2 : length(Ks)
            K = K + Ks{i};
        end
        %K = K(xmiss,xmiss);
        % Calculate the Score function
        yhat = cmpYhat(K,y);

        F = F + sum((y-yhat').^2);
        if nargout > 1
           % Compute gradient
           S = mlkrGrad(x,y,yhat',Kij,K,xmiss);
           D = vec(2*Ln*S);

           %% Constraint 
           D = reshape(D,d,d);
           %D = D + lambda*(D-LMean); % Not used...
           %D = diag(diag(D));
           %F = F + lambda*sum(sum(abs(D-LMean))); Failure... bad result.
           D = reshape(D,d*d,1);
           
           D_Final = D_Final + D;
        end
    end
   
    if nargin== 8 && ~isempty(fun), 
        fun(reshape(L,outdim,d));
    end; 
    D = D_Final;
end


function M = mlkrGrad(X,Y,yhat,Kij,K,xmiss) 
    N = size(Y,2);
    dy=bsxfun(@minus,repmat(yhat',1,N),Y);
    den = 1./(sum(K,2) - diag(K)); %<- here (maybe)the Kij should be calculated as the sum_p K_p 
    den(den==Inf)=1/eps;
    dd=(yhat-Y)';
    O = (bsxfun(@times,bsxfun(@times,dy,dd),den)+bsxfun(@times,bsxfun(@times,dy',dd'),den')); %This is the derivate in terms of Kij
    W=Kij.*O(xmiss,xmiss); % <- here the Kij should be calculated as the K_p 
    M=SODWm(X,W);    
end


function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end
