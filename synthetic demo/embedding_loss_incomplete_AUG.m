function [F,D] = embedding_loss_incomplete_AUG(w,X,Y,LFinal,W,w_ind,XMiss,AUG_Hidden)
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

    [d, ~ ,np] = size(X{w_ind}); np = np-1;
    [N,T] = size(Y);
    hidden_size = length(w)/d;
    AUG_Hidden_size = size(AUG_Hidden,1);
    w = reshape(w,hidden_size,d);
    P = ceil(T/np);
    F = 0;
    D_Final = zeros(d*hidden_size,1);
    
    x = zeros(hidden_size+AUG_Hidden_size, N, np); % hidden_size, # of locations, # of period
    W{w_ind} = w;
    for i = 1 : length(X)      
        for p = 1: size(X{i},3)
            x(:,XMiss{i},p) = [sigmoid(W{i}*X{i}(:,:,p)); AUG_Hidden(:,XMiss{i}) ];
        end
    end
        
    for tt = 2 : T
        tPeriod = ceil((tt+1)/P);
        y = Y(:,tt)';
        %Ln = reshape(LFinal, outdim, d);
        Lx = LFinal * x(:,:,tPeriod);
        Kij = cmpKij(Lx);
        yhat = cmpYhat(Kij,y);

        F = F + sum((y-yhat').^2);
        if nargout > 1
           % Compute gradient
           S = mlkrGrad2(X{w_ind}(:,:,tPeriod),y,yhat',Kij, w,XMiss{w_ind}, x(:,:,tPeriod),hidden_size, ((LFinal'*LFinal)+ (LFinal'*LFinal)')); % S: H*D
           D = vec( S);           
           D_Final = D_Final + D;
        end
    end
   

    D = D_Final;
end


function M = mlkrGrad2(X, Y, yhat, Kij,w,xmiss, fx, hidden_size, B)
    N = size(Y,2);
    dy=bsxfun(@minus,repmat(yhat',1,N),Y);
    den = 1./(sum(Kij,2) - diag(Kij));
    den(den==Inf)=1/eps;
    dd=(yhat-Y)';
    O = Kij.*(bsxfun(@times,bsxfun(@times,dy,dd),den)+bsxfun(@times,bsxfun(@times,dy',dd'),den'));
    
    M = zeros(size(w));

    fx_s = fx.*(1-fx);
    for i = xmiss'
        for j = 1: N
            fxij = B*(fx(:,i) -  fx(:,j));
            
            M = M + O(i,j)*( (fxij(1:hidden_size) .* fx_s(1:hidden_size,i))*X(:,xmiss==i)'  );
        end
    end
    
    for i = 1: N
        for j = xmiss'
            fxij = B*(fx(:,i) -  fx(:,j));
            M = M + O(i,j)*(  - fxij(1:hidden_size).* fx_s(1:hidden_size,j)*X(:,xmiss==j)' );
        end
    end
  
end


function [yhat] = cmpYhat(Kij, Y)
    Kijd = Kij - eye(size(Kij));
    su=sum(Kijd,1);
    su(su==0)=eps;
    yhat = (sum(bsxfun(@times,Kijd,Y'),1) ./ su)';
end
