function [F,D] = mmlkrLoss_global(L,X,Y,LMean,lambda,outdim,fun)
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
%
%
% copyright Jake Gardner and Kilian Q. Weinberger, 2012
%

    [d,N,np] = size(X);
    [N,T] = size(Y);
    p = ceil(T/np);
    if nargin<6,outdim=d;end;
    F = 0;
    D_Final = zeros(d*d,1);
    
    for tt = 2 : T
        tPeriod = ceil((tt)/p);
        x = X(:,:,tPeriod);
        y = Y(:,tt)';
        Ln = reshape(L, outdim, d);
        Lx = Ln * x;
        Kij = cmpKij(Lx);
        yhat = cmpYhat(Kij,y);

        F = F + sum((y-yhat').^2);
        if nargout > 1
           % Compute gradient
           S = mlkrGrad(x,y,yhat',Kij);
           D = vec(2*Ln*S);

           %% This part is changed.
           D = reshape(D,d,d);
           D = D + lambda*(D-LMean);
           %D = diag(diag(D));
           %F = F + lambda*sum(sum(abs(D-LMean))); Failure... bad result.
           D = reshape(D,d*d,1);
           
           D_Final = D_Final + D;
        end
    end
   
    if nargin==7 && ~isempty(fun), 
        fun(reshape(L,outdim,d));
    end;
    D = D_Final;
end


function M = mlkrGrad(X,Y,yhat,Kij)
    [d,N] = size(X);
     dy=bsxfun(@minus,repmat(yhat',1,N),Y);
    den = 1./(sum(Kij,2) - diag(Kij));
    den(den==Inf)=1/eps;
    dd=(yhat-Y)';
    W=Kij.*(bsxfun(@times,bsxfun(@times,dy,dd),den)+bsxfun(@times,bsxfun(@times,dy',dd'),den'));
    M=SODWm(X,W);    
end


function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end
