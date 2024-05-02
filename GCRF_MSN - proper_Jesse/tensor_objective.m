function [l, delta_theta] = tensor_objective(Theta,D,C,YLY,P,Ytot,RYtot)
%Thetas were log_thetas


[hospital_n,disease_n,t_train,acount] = size(C);

alpha = Theta(1:acount); 
beta = Theta(acount+1);
gamma = sum(alpha);

Lambda = (gamma + beta*D);
Lambda_inv = ones(hospital_n,disease_n) ./ Lambda; 

%Delta Alphas   outputs [1,a] so we can 

tr = sum(sum(Lambda_inv))*t_train; %Monthly Similarity for 9 Years

Ca = zeros(hospital_n,disease_n,t_train);
for i = 1:acount
    Ca = Ca + C(:,:,:,i)*alpha(i);
end

FunLambminsq = 0;
for j = 1:t_train
    FunLambminsq = FunLambminsq + sum(sum(Ca(:,:,j) .* Lambda_inv .* Lambda_inv .* Ca(:,:,j)));
    %\alpha'C'\Lambda^{-2}C\alpha
end

for i = 1:acount
       FunLambmininv = 0;
    for j = 1:t_train
        FunLambmininv =  FunLambmininv + sum(sum( C(:,:,j,i) .* Lambda_inv .* Ca(:,:,j) ));
        %C_i\Lambda^{-1}C\alpha
    end
    
delta_alpha(i) =   FunLambmininv;
end

delta_alpha = -0.5* Ytot + 2*RYtot + 0.5 * FunLambminsq -2* delta_alpha   + 0.25* tr ; 


%Delta Beta

tr = sum(sum(D .* Lambda_inv))*t_train; 

CaLiDCaLi = 0; %Ca' Lambda_inv D Lambda_inv Ca 
Lambda_inv2 = Lambda_inv .* Lambda_inv;
for j = 1:t_train
    CaLiDCaLi =  CaLiDCaLi ... 
            + sum(sum( Ca(:,:,j) .* Lambda_inv2 .* D.*  Ca(:,:,j) ));
end

delta_beta = CaLiDCaLi - YLY + 0.5*tr;

%Log Likelihood

Ca2 = Ca .* Ca;
yQytot=0;
muQmutot = 0; 
for j = 1:t_train
    yQytot = yQytot + sum(sum( P(:,:,j) .*  Lambda));
    muQmutot = muQmutot + sum(sum(Ca2(:,:,j) .* Lambda_inv ));
end

l = -yQytot - muQmutot + 2*alpha*RYtot' + 0.5 * t_train * sum(sum(log(Lambda)))+ 0.5* hospital_n*disease_n*log(2);
l=-l;
%disp('Log Likelihood l');

%disp(Lambda);
%disp(l);
%disp(-yQytot - muQmutot + 2*alpha*RYtot');
%disp(0.5 * t_train * sum(sum(log(Lambda)))+ 0.5* hospital_n*disease_n*log(2));
delta_theta = [-delta_alpha,-delta_beta];



