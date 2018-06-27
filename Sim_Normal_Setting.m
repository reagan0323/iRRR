% This file provides simulation settings for Gaussian data.


%% Simulation Setting
% basic setting: two-set, low-dim, ind-pred, sep-coef
n=500; 
num=2;
p1=50;
p2=50;
p=[p1,p2];
q=100;
r=10; % per coef mat rank

Gamma=eye(p1+p2); % covariance
hfGamma=Gamma^0.5; 

rng(123456)
L1=randn(p1,r);
L2=randn(p2,r);
R1=randn(q,r);
R2=randn(q,r);
B1=L1*R1';
B2=L2*R2';
B=[B1;B2];
missing=[]; % missing index for Y (12/10)

switch choosesetting
    case 11 % based on setting1, with 10% missing values
        missing=randsample(n*q,round(0.1*n*q)); % missing index of Y
    case 12 % 20% missing
        missing=randsample(n*q,round(0.2*n*q)); % missing index of Y
    case 13 % 30% missing
        missing=randsample(n*q,round(0.3*n*q)); % missing index of Y
    case 14 % 40% missing
        missing=randsample(n*q,round(0.4*n*q)); % missing index of Y

    
    case 2  % two-set, low-dim, CORR-PRED, sep-coef
        rho=0.9; % between var corr, across X1 and X2
        Gamma=ones(p1+p2)*rho+eye(p1+p2)*(1-rho); % override Gamma
        hfGamma=Gamma^0.5; % override hfGamma

    case 31  % two-set, low-dim, ind-pred, ONE-COEF
        r=20; % both [B1;B2] and B1 and B2 rank
        rng(123456)
        L1=randn(p1,r);
        L2=randn(p2,r);
        R=randn(q,r);
        B1=L1*R';
        B2=L2*R';
        B=[B1;B2]; % override B
        
    case 32  % two-set, low-dim, ind-pred, ONE-COEF
        r=40; % both [B1;B2] and B1 and B2 rank
        rng(123456)
        L1=randn(p1,r);
        L2=randn(p2,r);
        R=randn(q,r);
        B1=L1*R';
        B2=L2*R';
        B=[B1;B2]; % override B
        
    case 33  % two-set, low-dim, ind-pred, ONE-COEF
        r=60; % each B1 and B2 is full rank, b/c r>p1, r>p2
        rng(123456)
        L1=randn(p1,r);
        L2=randn(p2,r);
        R=randn(q,r);
        B1=L1*R';
        B2=L2*R';
        B=[B1;B2]; % override B
            
    case 41  % THREE-SET, low-dim, ind-pred, sep-coef
        num=3;
        p=ones(1,num)*p1; 

        B=[];
        Gamma=eye(sum(p));
        hfGamma=Gamma^0.5; 
        rng(123456)
        for i=1:num
            B=[B;randn(p1,r)*randn(r,q)];
        end;
    
    case 42  % FOUR-SET, low-dim, ind-pred, sep-coef
        num=4;
        p=ones(1,num)*p1; 

        B=[];
        Gamma=eye(sum(p));
        hfGamma=Gamma^0.5; 
        rng(123456)
        for i=1:num
            B=[B;randn(p1,r)*randn(r,q)];
        end;

    case 43  % FIVE-SET, low-dim, ind-pred, sep-coef
        num=5;
        p=ones(1,num)*p1; 

        B=[];
        Gamma=eye(sum(p));
        hfGamma=Gamma^0.5; 
        rng(123456)
        for i=1:num
            B=[B;randn(p1,r)*randn(r,q)];
        end;


    case 5  % THREE-SET, one is redundant
        num=3;
        p=ones(1,num)*p1; 

        Gamma=eye(sum(p));
        hfGamma=Gamma^0.5; 
        rng(123456)
        B=[B;zeros(p1,q)];
end;




%% Generate Tuning and Training predictors
% for aRRR, we have to use the same number of samples in both sets

% Tuning set
cX_tune=randn(n,sum(p))*hfGamma;
cX_tune=bsxfun(@minus, cX_tune, mean(cX_tune,1));
X_tune=cell(1,num);
tempp1=1+cumsum([0,p(1:(end-1))]);
tempp2=cumsum(p);
for i=1:num
    X_tune{i}=cX_tune(:,tempp1(i):tempp2(i));
end;


% Training set
cX=randn(n,sum(p))*hfGamma;
cX=bsxfun(@minus, cX, mean(cX,1));
X=cell(1,num);
for i=1:num
    X{i}=cX(:,tempp1(i):tempp2(i));
end;


%% adjust signal level in B
% std of E is fixed to be 1
temp=cX*B;  % linear predictor
c=quantile(abs(temp(:)),0.9);
Btrue=B/c; % set 90% quantile of linear predictor to be 1
Bcelltrue=cell(1,num);
for i=1:num
    Bcelltrue{i}=Btrue(tempp1(i):tempp2(i),:);
end;
Gammatrue=Gamma; % each Xi's covariance matrix


