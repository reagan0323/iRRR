% This file provides simulation settings for Binary outcome data.

%% Simulation Setting
% basic setting: two-set, low-dim, ind-pred, sep-coef
rng(123456)
switch choosesetting % determine rank
    case 1  % small sample size
        n_tune=200;% tuning and training
        n_train=200;  
        n_valid=500;% validating and testing
        n_test=500; 
        
        num=2;
        p1=50;
        p2=50;
        p=[p1,p2];
        q=100;
        r=10; % per coef mat rank
        tempp1=1+cumsum([0,p(1:(end-1))]);
        tempp2=cumsum(p);


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

        mu=unifrnd(-1,1,q,1); % uniform [-1,1]

    case 2  % with a redundant group
        n_tune=200;% tuning and training
        n_train=200;  
        n_valid=500;% validating and testing
        n_test=500; 
        
        num=3;
        p1=50;
        p2=50;
        p3=100;
        p=[p1,p2,p3];
        q=100;
        r=10; % rank per non-zero coef 
        tempp1=1+cumsum([0,p(1:(end-1))]);
        tempp2=cumsum(p);


        Gamma=diag([ones(p1+p2,1);ones(p3,1)*1]); % covariance, increase signal in X3 (does not matter for wiRRR)
        hfGamma=Gamma^0.5; 

        rng(123456)
        L1=randn(p1,r);
        L2=randn(p2,r);
        R1=randn(q,r);
        R2=randn(q,r);
        B1=L1*R1';
        B2=L2*R2';
        B3=zeros(p3,q);
        B=[B1;B2;B3];

        mu=unifrnd(-1,1,q,1); % uniform [-1,1]
end;


%% Generate Tuning, Validating, Training, Testing predictors
% Tuning set
cX_tune=randn(n_tune,sum(p))*hfGamma;
cX_tune=bsxfun(@minus, cX_tune, mean(cX_tune,1));
X_tune=cell(1,num);
for i=1:num
    X_tune{i}=cX_tune(:,tempp1(i):tempp2(i));
end;

% Validating set
cX_valid=randn(n_valid,sum(p))*hfGamma;
cX_valid=bsxfun(@minus, cX_valid, mean(cX_valid,1));
X_valid=cell(1,num);
for i=1:num
    X_valid{i}=cX_valid(:,tempp1(i):tempp2(i));
end;

% Training set
cX_train=randn(n_train,sum(p))*hfGamma;
cX_train=bsxfun(@minus, cX_train, mean(cX_train,1));
X_train=cell(1,num);
for i=1:num
    X_train{i}=cX_train(:,tempp1(i):tempp2(i));
end;

% Testing set
cX_test=randn(n_test,sum(p))*hfGamma;
cX_test=bsxfun(@minus, cX_test, mean(cX_test,1));
X_test=cell(1,num);
for i=1:num
    X_test{i}=cX_test(:,tempp1(i):tempp2(i));
end;

%% adjust signal level in B
temp=cX_train*B;  % linear predictor
c=quantile(abs(temp(:)),0.9);
Btrue=5*B/c; % set 90% quantile of linear predictor to be 5
Bcelltrue=cell(1,num);
for i=1:num
    Bcelltrue{i}=Btrue(tempp1(i):tempp2(i),:);
end;
Gammatrue=Gamma; % each Xi's covariance matrix
mutrue=mu;

