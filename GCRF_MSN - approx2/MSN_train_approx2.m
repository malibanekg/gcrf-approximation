function [alpha,beta,MSE_train_GCRF_IMPROVED2, output] = MSN_train_approx2(YMlist,Rlist,S1,S2)
%Generalized Tensor Train
%   Include Parallel Loop
[n1,~,timeversions1] = size(S1);
[n2,~,timeversions2] = size(S2);

[V,D1] = LaplacianApprox2(S1, S2);    

D = diag(D1);
D = reshape(real(D), n2,n1);
V=reshape(V,[n2,n1,n2,n1]);

% % Second part
% if timeversions1 > 1 
%     V_S = zeros(n1,n1,timeversions1);
%     D_S = zeros(n1,timeversions1);
%     parfor i = 1:timeversions1
%         D1 = sum(S1(i));  
%         D1 = D1.^(-.5);
%         S1_norm = S1(i) .* (D1 * D1'); 
%         [V_St,D_St] = eig(S1_norm); 
%         D_St = diag(D_St);
%         V_S(:,:,i) = V_St;  D_S(:,i) = D_St;
%     end
% else
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%     D1 = sum(S1);
%     D1 = D1.^(-.5);
%     S1_norm = S1 .* (D1' * D1);
%     [V_S,D_S] = eig(S1_norm); 
%     D_S = diag(D_S);
%     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% end  
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% D2 = sum(S2);  
% D2 = D2.^(-.5);
% S2_norm = S2 .* (D2' * D2); 
% [V_H,D_H] = eig(S2_norm); 
% D_H = diag(D_H);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % D1 = diag(D1);
% % D2 = diag(D2);
% 
% D = D_H * D_S';
% D = ones(n2,n1) - D;
% D = (D2' * D1) .* D;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[hospital_n,disease_n,t_train,acount] = size(Rlist); 
%[~,~,t_test,~] = size(RlistTest); 

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
                % Vcurrent = reshape((V_S(:,k) * V_H(:,l)')',[n2,n1]);
                Vcurrent= V(:,:,l,k);
                MatC(l,k) = sum(sum( Vcurrent  .* Rlist(:,:,i,j) ));%column of C coressponded to the eigenvector U_{k,l}
                if computed_UtY == 0 
                    UtY(l,k) = sum(sum( Vcurrent .* YMlist(:,:,i) ));%U_{k,l}'y_i, preprocessing for delta_beta 
                                                                     %and objective function
                end
            end
        end
        CMparallel{i,j} = MatC;
        computed_UtY=1;
        RY(j,i) = sum(sum(Rlist(:,:,i,j) .* YMlist(:,:,i)));%R_j^i'*y_i, preprocessing for delta_alpha
                                                            %and objective function  
    end  
         
    Ytot = Ytot + sum(sum(YMlist(:,:,i) .* YMlist(:,:,i))); % total y'y, preprocessing for delta_alpha
      
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

[Thetatest,fval, exitflag,output] = fmincon(@(Theta)tensor_objective(Theta,D,CMlist,YLYtot,Plist,Ytot,RYtot),Theta0,A,b,[],[],[],[],[],options);%

alpha = Thetatest(1:acount); beta = Thetatest(acount+1);

% Calculate train error
t = 1;
k = numel(alpha);
n = n1 * n2;

gamma = sum(alpha);
S = kron(S1,S2);
% S = S/sum(sum(S));
Q = gamma*eye(n)+ beta*(diag(sum(S)) - S);
clear S;
% I = eye(n);
% Inv = I + (I-Q) + (I-Q)^2 + (I-Q)^3 + (I-Q)^4 + (I-Q)^5 + (I-Q)^6 +(I-Q)^7 +(I-Q)^8;

% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha(j)*Rlist(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf; 
end

MSE_train_GCRF_IMPROVED2 = (reshape(YMlist,[n,1]) - mu)' * (reshape(YMlist,[n,1]) - mu)/n;

end
%%
%Yhat = 0;
%[Yhat] = MSN_predict();

