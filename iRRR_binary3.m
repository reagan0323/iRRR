function [C,mu,A,B,Theta]=iRRR_binary3(Y,X,lam1,paramstruct)
% This function uses consensus ADMM to fit the iRRR model. It is suitable
% for binary outcomes with potentially missing entries.
%
% Model:
% -4/n*sum[logL(Y_{non-missing})]  + lam1*sum(w_i*|A_i|_*) (+ 0.5*lam0*sum(w_i^2*|B_i|^2_F))
% s.t.  A_i=B_i
%
% The -4/n*sum[logL(Y_{non-missing})] can be majorized to
% 1/(2n)|Ystar-1*mu'-sum(Xi*Bi)|^2_F (this involves two steps of
% majorization: first majorize the log likelihood of each observed entry of
% Y; second majorize the missing entries with the current estimation)
%
% 
%
% input: 
%
%   Y       n*q binary response data matrix (may have nan entries)
%
%   X       1*K cell array, each cell is a n*p_i predictor data matrix
%           Note: X1,...XK may need some sort of standardization, because
%           we use a single lam0 and lam1 for different predictor sets.
%           Namely, we implicitly assume the coefficients are comparable.
% 
%   lam1    positive scalar, tuning for nuclear norm 
%
%   paramstruct
%           lam0    tuning for ridge penalty, default is 0
%   
%          weight   K*1 weight vector, default: a vector of 1; 
%                   By theory, we should use
%                   w(i)=(1/n)*max(svd(X{i}))*[sqrt(q)+sqrt(rank(X{i}))]; where
%                   X's are demeaned.
%                   Hueristically, one could also use w(i)=|X_i|_F
%
%          randomstart         0=false (default); 1=true
%
%          varyrho  0=fixed rho (default); 1=adaptive rho
%          maxrho   5 (default): max rho. Unused if varyrho==0
%
%          rho      step size, default rho=0.1
%
%          Tol      default 1E-3, 
%
%          Niter	default 500
%
%          fig      1 (default) show checking figures; 0 no show
%
% Output: 
%
%   C       sum(p_i)*q coefficient matrix, potentially low-rank
%
%   mu      q*1 intercept vector, for original X (not centered X)
%
%   A       cell arrays of length K, separate low-rank coefficient matrices
%
%   B       cell arrays of length K, separate coefficient matrices
%
%   Theta   cell arrays of length K, Lagrange parameter matrices
%
%
% Modified from iRRR_binary2 on 10/12/2017 by Gen Li
%   Note: treat ridge penalty as optional



% default parameters
K=length(X);
weight=ones(K,1);
Tol=1E-3; % stopping rule
Niter=500; % max iterations
lam0=0;
varyrho=0;
rho=0.1;
maxrho=5;
randomstart=0;
fig=1;
if nargin > 3 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'lam0') ;   
    lam0= paramstruct.lam0 ; 
  end ;
  if isfield(paramstruct,'weight') ;   
    weight = paramstruct.weight ; 
  end ;
  if isfield(paramstruct,'Tol') ;   
    Tol = paramstruct.Tol ; 
  end ;
  if isfield(paramstruct,'Niter') ;   
    Niter = paramstruct.Niter ; 
  end ;
  if isfield(paramstruct,'randomstart') ;   
    randomstart = paramstruct.randomstart ; 
  end ;
  if isfield(paramstruct,'varyrho') ;   
    varyrho = paramstruct.varyrho; 
  end ;
  if varyrho && isfield(paramstruct,'maxrho') ;
      maxrho = paramstruct.maxrho; 
  end;
  if isfield(paramstruct,'rho') ;   
    rho = paramstruct.rho; 
  end ;
  if isfield(paramstruct,'fig') ;   
    fig = paramstruct.fig ; 
  end ;
end;



[n,q]=size(Y);
missing=isnan(Y);
p=zeros(K,1);
cX=[]; % horizontally concatenated X
meanX=[];
for i=1:K
    [n_,p(i)]=size(X{i});
    if n_~=n
        error('Samples do not match!')
    end;
    % first center X
    meanX=[meanX,mean(X{i},1)];
    X{i}=bsxfun(@minus,X{i},mean(X{i},1));
    % second, normalize centered X{i}'s
    X{i}=X{i}/weight(i);
    cX=[cX,X{i}]; % column centered X
end;





% initial values 
mu=zeros(q,1); % intercept
B=cell(K,1); 
Theta=cell(K,1); % Lagrange params for B
cB=zeros(sum(p),q);% vertically concatenated B
cTheta=zeros(sum(p),q);
if randomstart
    cB=randn(sum(p),q);
    mu=randn(q,1);
else
    for j=1:q
        ind=~isnan(Y(:,j));
        [temp,info]=lassoglm(cX(ind,:),Y(ind,j),'binomial','Alpha',0.05,'Lambda',0.1);
        mu(j)=info.Intercept;
        cB(:,j)=temp;
    end;
end;
for i=1:K    
    B{i}=cB((sum(p(1:(i-1)))+1):sum(p(1:i)),:);
    Theta{i}=cTheta((sum(p(1:(i-1)))+1):sum(p(1:i)),:);
end;
A=B; % low-rank alias
cA=cB; 

%
[~,D_cX,V_cX]=svd((1/sqrt(n))*cX,'econ');
if ~varyrho % fixed rho
    DeltaMat=V_cX*diag(1./(diag(D_cX).^2+lam0+rho))*V_cX'+...
        (eye(sum(p))-V_cX*V_cX')/(lam0+rho);   % inv(1/n*X'X+(lam0+rho)I)
end;





% check obj value
obj=ObjValue1(Y,X,A,mu,lam0,lam1); % full objective function (with penalties)
obj_ls=ObjValue1(Y,X,A,mu,0,0); % only the least square part






%%%%%%%%%%%%%%%
% MM + ADMM(one-step)
niter=0;
diff=inf;
rec_obj=[obj;obj_ls]; % record obj value
rec_Theta=[]; % Fro norm of Theta{1}
rec_primal=[]; % record total primal residual
rec_dual=[]; % record total dual residual
while niter<Niter  && abs(diff)>Tol
    niter=niter+1;
    cB_old=cB;
    
    %%%%%%%%%%%%% Double Majorization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Eta = ones(n,1)*mu' + cX*cB; % current linear predictor
    wY = Eta + 4*(2*Y-1).*(1-invlogit((2*Y-1).*Eta)); % working response for induced LS
    wY(missing) = Eta(missing);% majorize to get rid of missing
    mu=mean(wY,1)'; % new est of mu, b/c cX is col centered
    wY1=bsxfun(@minus,wY,mu'); 
    
  
    %%%%%%%%%%%%% ADMM(one-step) part %%%%%%%%%%%%%%%%%%%%
    % est B
    if varyrho
        DeltaMat=V_cX*diag(1./(diag(D_cX).^2+lam0+rho))*V_cX'+...
            (eye(sum(p))-V_cX*V_cX')/(lam0+rho); 
    end;
    cB=DeltaMat*((1/n)*cX'*wY1+rho*cA+cTheta);
    for i=1:K
        B{i}=cB((sum(p(1:(i-1)))+1):sum(p(1:i)),:);
    end;   
   
              
    % est A in parallel
    % update Theta
    parfor i=1:K
        % est A
        temp=B{i}-Theta{i}/rho;
        [tempU,tempD,tempV]=svd(temp,'econ');
        A{i}=tempU*SoftThres(tempD,lam1/rho)*tempV';
        
        % update Theta
        Theta{i}=Theta{i}+rho*(A{i}-B{i});
    end;
    % reshape cA and cTheta
    for i=1:K
        cA((sum(p(1:(i-1)))+1):sum(p(1:i)),:)=A{i};
        cTheta((sum(p(1:(i-1)))+1):sum(p(1:i)),:)=Theta{i};
    end;


    
     % update rho
    if varyrho
        rho=min(maxrho,1.1*rho); % steadily increasing rho
    end;
    

    
    %%%%%%%%%%%%%%%% stopping rule %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % primal and dual residuals
    primal=norm(cA-cB,'fro')^2;
    rec_primal=[rec_primal,primal];
    dual=norm(cB-cB_old,'fro')^2;
    rec_dual=[rec_dual,dual];  

    % objective function value
    obj=ObjValue1(Y,X,A,mu,lam0,lam1);
    obj_ls=ObjValue1(Y,X,A,mu,0,0);
    rec_obj=[rec_obj,[obj;obj_ls]];
    
    % stopping rule
    diff=primal; 
%     diff=dual;
%     diff=rec_obj(1,end-1)-rec_obj(1,end);




    % Check Figures
    if fig==1
        % obj fcn values
        figure(101);clf; 
        plot(0:niter,rec_obj(1,:),'bo-');
        hold on
        plot(0:niter,rec_obj(2,:),'ro-');
        legend('Full Obj Value (with penalty)','Nagetive Likelihood Value')
        title(['Objective function value (decrease in full=',num2str(rec_obj(1,end-1)-rec_obj(1,end)),')']);
        drawnow;

        % primal and dual residuals
        figure(102);clf;
        subplot(1,2,1)
        plot(1:niter,rec_primal,'o-');
        title(['Primal residual |A-B|^2: ',num2str(primal)]);
        subplot(1,2,2)
        plot(1:niter,rec_dual,'o-');
        title(['Dual residual |B-B|^2: ',num2str(dual)]);
        drawnow
    
        figure(103);clf;
        rec_Theta=[rec_Theta,norm(Theta{1},'fro')];
        plot(rec_Theta,'o-');
        title('Theta: Lagrange multiplier for B1');
        drawnow

    end;



end;


if niter==Niter
    disp(['iRRR_binary does NOT converge after ',num2str(Niter),' iterations!']);
else
    disp(['iRRR_binary converges after ',num2str(niter),' iterations.']);      
end;

 
% output
% rescale parameter estimate, and add back mean
C=[];
for i=1:K
    A{i}=A{i}/weight(i);
    B{i}=B{i}/weight(i);
    C=[C;A{i}];
end;
clear cA cB;
mu=mu-(meanX*C)'; % convert to the original scale, so that mu+X_original*C is the final linear predictor

end



function Dout=SoftThres(Din,lam)
% this function soft thresholds the diagonal values of Din
% Din is a diagonal matrix
% lam is a positive threshold
% Dout is also a diagonal matrix
d=diag(Din);
d(d>0)=max(d(d>0)-lam,0);
d(d<0)=min(d(d<0)+lam,0);
Dout=diag(d);
end



function obj=ObjValue1(Y,X,B,mu,lam0,lam1) % for binary response
% (-4/n)logL(nonmissing) + penalty
% linear predictor is 1*mu'+XB
[n,~]=size(Y);
K=length(X);
obj=0;
pred=ones(n,1)*mu'; % linear predictor
for i=1:K
    pred=pred+X{i}*B{i}; % 
    obj=obj+lam0/2*norm(B{i},'fro')^2+lam1*sum(svd(B{i}));
end;
core=(2*Y-1).*pred;
neg4loglik=-(4/n)*sum(sum(log(invlogit(core)),'omitnan'),'omitnan');  % (-4/n)logL(nonmissing)
obj=obj +neg4loglik;
end

function out=invlogit(in)
out=exp(in)./(1+exp(in));
end
