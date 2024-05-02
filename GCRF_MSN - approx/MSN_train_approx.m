function [alpha,beta,MSE_train_GCRF_IMPROVED, output] = MSN_train_approx(Ytrain,Rtrain,S1,S2)
%Generalized Tensor Train
%   Include Parallel Loop

[n1,~,timeversions1] = size(S1);
[n2,~,timeversions2] = size(S2);
% S1 = S1/sum(sum(S1));
% S2 = S2/sum(sum(S2));

[V,D1] = LaplacianApprox(S1, S2); 
% [D1, indy] = sort(diag(D1));
% D1 = diag(D1);
% V = V(:, indy);
% indices = (V > 0 & V < 1e-15) | (V < 0 & V > -1e-15);
% V(indices) = 0;

% S = kron (S1,S2);
% Deg = sum(S);
% Deg = diag(Deg);
% Lap = Deg - S;
% [V,D1] = eig(Lap);

D = diag(D1);
% disp('========');
% disp('========');
D =reshape(D, n2,n1);

%Res=reshape(V,[n1,n2,n1,n2]);
%V=permute(Res,[2,1,4,3]);

V=reshape(V,[n2,n1,n2,n1]);
%V=permute(Res,[1,2,4,3]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[hospital_n,disease_n,t_train,acount] = size(Rtrain); 
%[~,~,t_test,~] = size(RtrainTest); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ytot = 0; YLYtot = 0; 

CMlist = zeros(hospital_n,disease_n,t_train,acount);

CMparallel = cell(t_train,acount);

for i = 1:t_train
    computed_UtY = 0; 
    UtY=zeros(hospital_n,disease_n); % needed for parfor
    for j = 1:acount
        MatC=zeros(hospital_n,disease_n); %needed for parfor
        for k = 1:disease_n
            for l = 1:hospital_n
                %Vcurrent = V_H(:,k) * V_S(:,l)';
                Vcurrent= V(:,:,l,k);
                MatC(l,k) = sum(sum( Vcurrent  .* Rtrain(:,:,i,j) ));%column of C coressponded to the eigenvector U_{k,l}
                if computed_UtY == 0 
                    UtY(l,k) = sum(sum( Vcurrent .* Ytrain(:,:,i) ));%U_{k,l}'y_i, preprocessing for delta_beta 
                                                                     %and objective function
                end
            end
        end
        CMparallel{i,j} = MatC;
        computed_UtY=1;
        RY(j,i) = sum(sum(Rtrain(:,:,i,j) .* Ytrain(:,:,i)));%R_j^i'*y_i, preprocessing for delta_alpha
                                                            %and objective function  
    end  
         
    Ytot = Ytot + sum(sum(Ytrain(:,:,i) .* Ytrain(:,:,i))); % total y'y, preprocessing for delta_alpha
      
    YLYtot = YLYtot + sum(sum(UtY.^2 .* D)); %total y'UDU'y,  preprocessing for delta_beta
   
    Plist(:,:,i) = UtY .*UtY; % preprocessing for Log Likelihood
    
end

for i = 1:t_train  
    for j = 1:acount
        CMlist(:,:,i,j) = CMparallel{i,j};
    end
end
clear CMparallel;

RYtot = sum(RY,2)'; 

%CM 10 seconds per t => 720 seconds
%Entire Process 20 seconds per t => 1440 seconds

%%

%initialize 
alpha0 = ones(1,acount); beta0 = 0; Theta0 = [alpha0,beta0];
%Theta0 = [0.4, 0.6,0.5];
dmax = max(max(D)); dmin = min(min(D));
A = [ones(1,acount) , dmax ; ...
     ones(1,acount) , dmin]; 
A = -A;
b = [0; 0];

%Run Gradient
options = optimset('Algorithm','interior-point', 'Display','off','MaxIter',200,'GradObj','on','TolX',1e-100,'TolCon',1e-50); %optimization
%options = optimset(options,'UseParallel','always');

[Thetatest,fval,exitflag,output] = fmincon(@(Theta)tensor_objective(Theta,D,CMlist,YLYtot,Plist,Ytot,RYtot),Theta0,A,b,[],[],[],[],[],options);%

alpha = Thetatest(1:acount); beta = Thetatest(acount+1);

% Calculate train error
t = 1;
k = numel(alpha);
n = n1 * n2;

gamma = sum(alpha);
S = kron(S1,S2);
Q = gamma*eye(n)+ beta*(diag(sum(S)) - S);
clear S;
% I = eye(n);
% Inv = I + (I-Q) + (I-Q)^2 + (I-Q)^3 + (I-Q)^4 + (I-Q)^5 + (I-Q)^6 +(I-Q)^7 +(I-Q)^8;

% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha(j)*Rtrain(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf; 
end

MSE_train_GCRF_IMPROVED = (reshape(Ytrain,[n,1]) - mu)' * (reshape(Ytrain,[n,1]) - mu)/n;

end
%%
%Yhat = 0;
%[Yhat] = MSN_predict();

