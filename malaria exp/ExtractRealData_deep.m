function [Data] = ExtractRealData_deep(beginyear,endyear,MissingRate,MissingMode,FeatureType)
%EXTRACTREALDATA Summary of this function goes here
%   Detailed explanation goes here
% PARAM FeatureType: 1:All; 2: Manually Selected; 3: My way for selection;
% 4: PCA;

load('data/Case_Town62.mat');%Case 7x1 cells
load('data/Location.mat');%Loc 62x2
load('data/VCAP_Town62.mat');%VCAP 62x161 (23*7)
load('data/EconomicStatus.mat');%EconomicStatus 18x154 (22*7)
load('data/elevation.mat');%elevation 62x1
load('data/pop.mat');%pop 62x1
load('data/MODIS_TRMM_Town62.mat'); %MODIS TRMM
temp = [];
for i = 3 : length(Case)
    temp = [ temp Case{i}];
end
Case = temp;
VCAP = VCAP(:,23*2+1:end);
MODIS = MODIS(:,23*2+1:end);
TRMM = TRMM(:,23*2+1:end);
EconomicStatus = EconomicStatus(:,22*2 +1:end);

T = 11; 
p = 22; % feature length of the economic data
n = 62; % number of location


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Semi-month to month 23 INTO 12%%%%%%%%%%%%%%%%%%
IGNORE = (23:23:size(Case,2)); % Igore the last interval
Case(:,IGNORE) = [];
VCAP(:,IGNORE) = []; % Become 22 Interval
MODIS(:,IGNORE) = []; % Become 22 Interval
TRMM(:,IGNORE) = []; % Become 22 Interval

ODD = (1:2:size(Case,2));
EVEN = (2:2:size(Case,2));

Case = (Case(:,ODD) + Case(:,EVEN));
VCAP = (VCAP(:,ODD) + VCAP(:,EVEN));
MODIS = (MODIS(:,ODD) + MODIS(:,EVEN));
TRMM = (TRMM(:,ODD) + TRMM(:,EVEN));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TimeIntYear = endyear - beginyear;
Case = Case(:,beginyear*T+1:endyear*T); % 10-12
VCAP = VCAP(:,beginyear*T+1:endyear*T);
MODIS = MODIS(:,beginyear*T+1:endyear*T);
TRMM = TRMM(:,beginyear*T+1:endyear*T);

EconomicStatus = EconomicStatus(:,beginyear*p+1:endyear*p);

Data.NPeriod = TimeIntYear;
Data.T = T;
Data.Location = Loc;

%% 
Data.X = {};
Data.XMiss = {};

temp = [ones(20,1);zeros(42,1)];
temp([1,4]) = 0;
Data.XMiss{1} = find(temp==1);
Data.XMiss{2} = find(temp==0);

% Data 1
X1 = zeros(24,18,TimeIntYear);
for ii = 1 : TimeIntYear
    X1(1:22,:,ii) = EconomicStatus(:,1+(ii-1)*22:ii*22)';
    X1(23:24,:,ii) = Loc(Data.XMiss{1},:)';
end
%X1 = log(X1);
%X1(X1 < 0) = 0;
Data.X{1} = min_max(X1); 


% Data 2
X2 = zeros(2,n-18,TimeIntYear);
for ii = 1 : TimeIntYear
    X2(:,:,ii) = Loc(Data.XMiss{2},:)';
end

%X2 = log(X2);
%X2(X2 < 0) = 0;

Data.X{2} = min_max(X2); 




%% Min-Max normalize
% for ii = 1 : TimeIntYear
%     for jj = 1 : p        
%         Data.X(jj,:,ii) = (Data.X(jj,:,ii)-min( Data.X(jj,:,ii)))/(max( Data.X(jj,:,ii))- min(Data.X(jj,:,ii)));     
%     end
% end



%% Symbol Representation for the Covariance Learning
Data.P = T;
Data.N = n;
Data.I = Case;
R = MissingModelComplete(Data,MissingMode,MissingRate);

TForTrain = (TimeIntYear -1)*T;
Data.RForTest = R(:,TForTrain+1:end);
Data.RForTest(:,[1,2]) = 1;
Data.IForTest = Case(:,TForTrain+1:end);
Data.RForTrain = R(:,1:TForTrain);
Data.ssForTest = VCAP(:,TForTrain+1:end);
Data.I = Case(:,1:TForTrain);
Data.R = R(:,1:TForTrain);
Data.ss = VCAP(:,1:TForTrain);

TemporalX = [VCAP;TRMM;MODIS];
Data.TemporalX_ForTrain = TemporalX(:,1:TForTrain);
Data.TemporalX_ForTest = TemporalX(:,TForTrain+1:end);
end



function X = min_max(X)
[D,N,T] = size(X);
for t = 1 : T
    for d = 1 : D
        max_value = max(X(d,:,t));
        min_value = min(X(d,:,t));
        X(d,:,t) = (X(d,:,t) - min_value)/(max_value-min_value);
    end
end
end
