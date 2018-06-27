% This file runs simulation studies for the normal cases.


clear all;
close all;
clc
ppth=''; % put directory here


choosesetting=1; % choose from [1, 2, 31:33, 41:43, 5]
Sim_Normal_Setting


% identity error covariance
SigmaE=eye(q);
SigmaEhf=eye(q);

% AR(1) error covariance matrix
SigmaE=toeplitz(0.5.^(0:(q-1))); 
SigmaEhf=SigmaE^.5;


%% %%%%%%%%%%%%%% Select Tuning Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using tuning data
% and just once, no repetition  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate tuning set
E_tune=randn(n,q)*SigmaEhf;
E_tune=bsxfun(@minus,E_tune,mean(E_tune,1));
Y_tune=cX_tune*Btrue+E_tune;

% make response missing (12/10)
Y_tune(missing)=NaN;


% candidate sets of tuning parameters
cand_iRRR_lam1=10.^[-1.5:.1:0.5]; % nuclear




%% iRRR (theoretical w_i)
n2=length(cand_iRRR_lam1);

% weight
weight=[];
for idata=1:length(X_tune)
    weight=[weight,max(svd(X_tune{idata}))*(sqrt(q)+sqrt(rank(X_tune{idata})))/size(X_tune{idata},1)];
end;
iRRR_out=zeros(1,n2);
selrank=iRRR_out;
for j=1:n2;
    lam1=cand_iRRR_lam1(j);
    [Bout,mu,Bcell_out,~,~]=iRRR_normal3(Y_tune,X_tune,lam1,...
        struct('varyrho',1,'Tol',0.01,'fig',0,'weight',weight));
    iRRR_out(j)=trace((Btrue-Bout)'*Gammatrue*(Btrue-Bout));
    selrank(j)=rank(Bout);
end;
[MSE1,ind2]=min(iRRR_out);
lam1_wiRRR=cand_iRRR_lam1(ind2);
[min(cand_iRRR_lam1),lam1_wiRRR,max(cand_iRRR_lam1)] 

% check if cand range covers the optimal
figure(1);clf;
plot(log10(cand_iRRR_lam1),iRRR_out);
ylabel('PMSE')
title(['Weighted iRRR lam1 Selection']);
xlabel('log10(iRRR lam1 range)');
orient landscape
print('-dpdf',[ppth,'Normal_Tune_',num2str(choosesetting),'_wiRRR']);



%% repeated simulation runs

% calc weight for iRRR
weight=[];
for idata=1:num
    weight=[weight,max(svd(X{idata}))*(sqrt(q)+sqrt(rank(X{idata})))/n];
end;

% multiple simulation run
nsim=100;
rec_est=zeros(nsim,1); 
rec_pred=zeros(nsim,1);
rec_rank=zeros(nsim,1); 
rec_seprank=zeros(nsim,num);
rec_nuclear=zeros(nsim,1); 
rec_time=zeros(nsim,1); 

rng(13579)
for isim=1:nsim
    disp(['Running Sim_',num2str(isim)]);
    
    % simulate data in each simulation run
    E=randn(n,q)*SigmaEhf;
    E=bsxfun(@minus,E,mean(E,1));
    Y=cX*Btrue+E;
    Y(missing)=NaN;
    
    % implement diff methods
    % iRRR
    time1=tic;
    [B_wiRRR,~,Bcell_wiRRR,~,~]=iRRR_normal3(Y,X,lam1_wiRRR,...
        struct('varyrho',1,'Tol',0.01,'fig',0,'weight',weight));    
    T1=toc(time1);

    
    % evaluate est
    rec_est(isim,:)=norm(B_wiRRR-Btrue,'fro');
    rec_pred(isim,:)=trace((B_wiRRR-Btrue)'*Gammatrue*(B_wiRRR-Btrue));
    rec_rank(isim,:)=rank(B_wiRRR);
    temprank=zeros(1,num);
    for i=1:num
        temprank(i)=rank(Bcell_wiRRR{i});
    end;
    rec_seprank(isim,:)=temprank;
    rec_nuclear(isim,:)=sum(svd(B_wiRRR,'econ'));
    rec_time(isim,:)=T1;
    

  
end;

mean(rec_pred(:,1))
std(rec_pred(:,1))
mean(rec_time(:,1))
std(rec_time(:,1))

