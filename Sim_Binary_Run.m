% This file conducts simulation studies for the binary cases.
clear all;
close all;
clc
ppth=''; % work directory

%% obtain sim setting
choosesetting=1; % choose from [1, 2]
Sim_Binary_Setting

%% %%%%%%%%%%%%%% Select Tuning Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using tuning data
% and just once, no repetition  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generate tuning set
Theta_tune=ones(n_tune,1)*mutrue'+cX_tune*Btrue; % n*q
Pi_tune=exp(Theta_tune)./(1+exp(Theta_tune));
Y_tune=binornd(1,Pi_tune);

% generate validating data 
Theta_valid=ones(n_valid,1)*mutrue'+cX_valid*Btrue; % n_test*q
Pi_valid=exp(Theta_valid)./(1+exp(Theta_valid));


% tuning for iRRR
cand_iRRR_lam1=10.^[-1.5:0.1:0.5]; % nuclear norm for iRRR




%% iRRR
n2=length(cand_iRRR_lam1);
% weight
weight=[];
for idata=1:num
    weight=[weight,max(svd(X_tune{idata}))*(sqrt(q)+sqrt(rank(X_tune{idata})))/n_tune];
end;
iRRR_out=zeros(1,n2);
for j=1:n2;
    lam1=cand_iRRR_lam1(j);
    [B_iRRR,mu_iRRR,~,~,~]=iRRR_binary3(Y_tune,X_tune,lam1,...
        struct('varyrho',1,'Tol',0.01,'fig',0,'weight',weight));
    Theta_est=ones(n_valid,1)*mu_iRRR'+cX_valid*B_iRRR;
    Pi_est=exp(Theta_est)./(1+exp(Theta_est));
    iRRR_out(j)=-sum(sum(Pi_valid.*log(Pi_est)+(1-Pi_valid).*log(1-Pi_est),'omitnan'),'omitnan'); % cross entropy of validating data
end;
[DEV_wiRRR,ind2]=min(iRRR_out);
lam1_wiRRR=cand_iRRR_lam1(ind2);
%
[min(cand_iRRR_lam1),lam1_wiRRR,max(cand_iRRR_lam1)] 

% check if cand range covers the optimal
figure(1);clf;
plot(log10(cand_iRRR_lam1),iRRR_out);
ylabel('Cross Entropy')
title(['Weighted iRRR lam1 Selection']);
xlabel('log10(wiRRR lam1 range)');
orient landscape
print('-dpdf',[ppth,'Binary_Tune_',num2str(choosesetting),'_wiRRR']);










%% %%%%%%%%%%%%%%% Repeated Simulation Runs %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fixed training and testing data (deterministic)
% training data
Theta_train=ones(n_train,1)*mutrue' + cX_train*Btrue; % n*q
Pi_train=exp(Theta_train)./(1+exp(Theta_train));
% testing data
Theta_test=ones(n_test,1)*mutrue' + cX_test*Btrue; % n_test*q
Pi_test=exp(Theta_test)./(1+exp(Theta_test));


% calc weight
weight=[];
for idata=1:num
    weight=[weight,max(svd(X_train{idata}))*(sqrt(q)+sqrt(rank(X_train{idata})))/n_train];
end;




% multiple simulation run
nsim=100;
rec_estB=zeros(nsim,1); 
rec_estMu=zeros(nsim,1); 
rec_pred=zeros(nsim,1); 
rec_rank=zeros(nsim,1);
rec_seprank=zeros(nsim,num); 
rec_nuclear=zeros(nsim,1); 
rec_time=zeros(nsim,1); 

rng(13579) % set seed
for isim=1:nsim
    disp(['Running Sim_',num2str(isim)]);
    % simulate training data in this run
    Y_train=binornd(1,Pi_train);


    % iRRR
    time0=tic;
    [B_wiRRR,mu_wiRRR,Bcell_wiRRR,~,~]=iRRR_binary3(Y_train,X_train,lam1_wiRRR,...
        struct('varyrho',1,'Tol',0.01,'fig',0,'weight',weight));
    T0=toc(time0);
    estTheta=ones(n_test,1)*mu_wiRRR'+cX_test*B_wiRRR;
    estPi=exp(estTheta)./(1+exp(estTheta));
    DEV_wiRRR=-sum(sum(Pi_test.*log(estPi)+(1-Pi_test).*log(1-estPi)));

    
    
    
    % evaluate est
    rec_estB(isim,:)=norm(B_wiRRR-Btrue,'fro');
    rec_estMu(isim,:)=norm(mu_wiRRR-mutrue,'fro');
    rec_pred(isim,:)=DEV_wiRRR; 
    temprank=zeros(1,num);
    for i=1:num
        temprank(i)=rank(Bcell_wiRRR{i});
    end;
    rec_seprank(isim,:)=temprank;
    rec_rank(isim,:)=rank(B_wiRRR);
    rec_nuclear(isim,:)=sum(svd(B_wiRRR,'econ'));
    rec_time(isim,:)=T0; 


    
    
end;

mean(rec_pred(:,1))
std(rec_pred(:,1))
mean(rec_time(:,1))
std(rec_time(:,1))

