function[Yhat] = MSN_predict()


%Need For Fit Evaluation
%Prediction

%Yhat = U InverseLam C alpha; 
% Below Set Up is Currently, U^T 
%   it might be that before was wrong
%   

%{
gamma = sum(alpha);
D2 = ones(hospital_n,disease_n) - D;
M = 2*(gamma + beta*D2);
M_inv = ones(hospital_n,disease_n) ./ M; 
Ca = zeros(hospital_n,disease_n,t_test);
for i = 1:acount
    Ca = Ca + CMlistTest(:,:,:,i)*alpha(i);
end

Yhat = zeros(hospital_n,disease_n,t_test);
%disp('Calculate M_inv');
%disp('Calcluate Ca');
%for j = 1:t_test
tic
for j = 1:1 
    for k = 1:hospital_n
        for l = 1:disease_n
            Yhat(k,l,j) = sum(sum((V_H(k,:)' * V_S(l,:)) .* M_inv .* Ca(:,:,j))); 
        end
    end
end
disp('Prediction Time');
toc
MSE_te_GCRF = sum(sum(sum((Yhat - YMlistTest) .* (Yhat - YMlistTest) ))) / (t_test*disease_n*hospital_n);

%}