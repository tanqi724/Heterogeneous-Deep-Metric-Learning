function R = MissingModelComplete(Data,iMode,MissingRate)
% Here we divide locations into groups which contains 3 locations;
Y = Data.I;
P = Data.P;
N = Data.N;
NP = size(Y,2)/P;
R = ones(N,NP*P); %Observarion indicator: 1 for observed; 0 for unobserved.
iMode
switch (iMode)
    case 'MCAR' %MCAR
        disp('Complete Random Missing Generating')
        R = rand(size(Y));
        R = R > MissingRate;
        R(:,1) = 1;
        
    case 'MPS-MC'
        disp('Temporal Missing Missing Continuous Generating')
        R = ones(size(Y));
        for n = 1 : size(Y,1)
            for np = 1 : 1 : NP
                length_missing = ceil(MissingRate*P);
                begin_t = min(ceil(rand()*(1-MissingRate)*P)+1,P);
                end_t = min(begin_t+length_missing-1, P);
                R(n,begin_t + (np-1)*P : end_t+(np-1)*P) = 0;
            end
        end
    
     case 'MPS-OC'
        disp('Temporal Missing Observation Continuous Generating')
        
        R = zeros(size(Y));
        for n = 1 : size(Y,1)
            for np = 1 : 1 : NP
                length_obs = ceil((1-MissingRate)*P);
                begin_t = min(ceil(rand()*MissingRate*P)+1,P);
                end_t = min(begin_t+length_obs-1, P);
                R(n,begin_t + (np-1)*P : end_t+(np-1)*P) = 1;
            end
        end
         
    case 'MSS-RS'
        disp('Spatial Missing Random Site Generating')
        
        yearIndicator = rand(N,(NP-1)) > MissingRate;
        PyearIndicator = rand(N,1)>MissingRate;
        PyearIndicator(ceil(rand()*N)) = 0; %At least one
        R = [kron(yearIndicator,ones(1,P)) kron(PyearIndicator,ones(1,P))];
        R(:,(1:P:NP*P)) = ones(N,NP);
        R(:,(2:P:NP*P)) = ones(N,NP);
        
    case 'MSS_NS'
        disp('Spatial Missing with Neighborhood Site Generating')
        
        K = 2;
        R = zeros(N,NP);
        distance = pdist2(Data.Location,Data.Location);
        for year = 1 : NP
            missing_loc = ones(N,1);
            
            seed_location = randsample(N,ceil(MissingRate*(N/K)));
            missing_loc(seed_location) = 0;
            for j = 1 : length(seed_location)
                [~,temp_index] = sort(distance(seed_location(j),:),'ascend');
                missing_loc(temp_index(K)) = 0;
            end
            R(:,year) = missing_loc;
        end
       
        R = kron(R,ones(1,P));
        R(:,(1:P:NP*P)) = ones(N,NP);
        R(:,(2:P:NP*P)) = ones(N,NP);
        
    otherwise
        error('Missing Mode Wrong Code.............')
end


end