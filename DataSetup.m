function [BaseResults, ApproxResults, ApproxResults2, MSNResults] = DataSetup(S1, S2, Ytrain, Rtrain, S1t, S2t, Ytest, Rtest, noise1, noise2, noise12, noise22) 
% Ytrain, Rtrain, 
%% TRAIN SETTINGS
[n1,~] = size(S1);
[n2,~] = size(S2);
n = n1 * n2;

S1prim = zeros(n1,n1);
S2prim = zeros(n2,n2);

%% ------------------------------------------------------------------------ 
% GCRF train setup

t = 1;
beta = 5; 
alpha = 1;
k = numel(alpha);
gamma = sum(alpha);

Ytrain1_noise = Ytrain1 + noise1; % normrnd(0,1/3,[n2,1]);
Ytrain2_noise = Ytrain2 + noise2; % normrnd(0,1/3,[n2,1]);

for i = 1:n1-1
    for j = (i+1):n1
        if S1(i,j) == 1
            S1prim(i,j) = exp(-abs(Ytrain1_noise(i) - Ytrain1_noise(j))); % + normrnd(0,0.1);
            S1prim(j,i) = S1prim(i,j);
        else
            S1prim(i,j) = 0;
            S1prim(j,i) = S1prim(i,j);
        end
    end
end

for i = 1:n2-1
    for j = (i+1):n2
        if S2(i,j) == 1
            S2prim(i,j) = exp(-abs(Ytrain2_noise(i) - Ytrain2_noise(j))); % + normrnd(0,0.1);
            S2prim(j,i) = S2prim(i,j);
        else
            S2prim(i,j) = 0;
            S2prim(j,i) = S2prim(i,j);
        end
    end
end

%% TRAINING PHASE =============================================================================
% Baseline model
tic
[alpha_r,beta_r,MSE_train_GCRF_BASE, output] = MSN_train_base(Ytrain,Rtrain,S1prim,S2prim);
time_base = toc;
iter_base = output.iterations;
% alpha_true_b = alpha_r/sum(alpha_r);
% beta_true_b = beta_r/sum(alpha_r);
alpha_true_b = alpha_r;
beta_true_b = beta_r;

% Approximated model 
tic
[alpha_r,beta_r,MSE_train_GCRF_IMPROVED, output] = MSN_train_approx(Ytrain,Rtrain,S1prim,S2prim);
time_improved = toc;
iter_improved = output.iterations;
% alpha_true_a = alpha_r/sum(alpha_r);
% beta_true_a = beta_r/sum(alpha_r);
alpha_true_a = alpha_r;
beta_true_a = beta_r;

% Approximated model 2
tic
[alpha_r,beta_r,MSE_train_GCRF_IMPROVED2, output] = MSN_train_approx2(Ytrain,Rtrain,S1prim,S2prim);
time_improved2 = toc;
iter_improved2 = output.iterations;
% alpha_true_a2 = alpha_r/sum(alpha_r);
% beta_true_a2 = beta_r/sum(alpha_r);
alpha_true_a2 = alpha_r;
beta_true_a2 = beta_r;

% Jesse model
tic
[alpha_r,beta_r, MSE_train_GCRF_MSN, output] = MSN_train_jesse(Ytrain,Rtrain,S1prim,S2prim); 
time_msn = toc;
iter_msn = output.iterations;
% alpha_true_j = alpha_r/sum(alpha_r);
% beta_true_j = beta_r/sum(alpha_r);
alpha_true_j = alpha_r;
beta_true_j = beta_r;

%% TEST SETTINGS =============================================================================
rng('shuffle');

S1tprim = zeros(n1,n1);
S2tprim = zeros(n2,n2);

Ytest1_noise = Ytest1 + noise12; % normrnd(0,1/3,[n1,1]);
Ytest2_noise = Ytest2 + noise22; % normrnd(0,1/3,[n2,1]);

for i = 1:n1-1
    for j = (i+1):n1
        if S1t(i,j) == 1
            S1tprim(i,j) = exp(-abs(Ytest1_noise(i) - Ytest1_noise(j))); % + normrnd(0,0.1);
            S1tprim(j,i) = S1tprim(i,j);
        else
            S1tprim(i,j) = 0;
            S1tprim(j,i) = S1tprim(i,j);
        end
    end
end

for i = 1:n2-1
    for j = (i+1):n2
        if S2t(i,j) == 1
            S2tprim(i,j) = exp(-abs(Ytest2_noise(i) - Ytest2_noise(j))); % + normrnd(0,0.1);
            S2tprim(j,i) = S2tprim(i,j);
        else
            S2tprim(i,j) = 0;
            S2tprim(j,i) = S2tprim(i,j);
        end
    end
end

S = kron(S1tprim,S2tprim);
LS = diag(sum(S)) - S;
clearvars S;


%% CALCULATE ACCURACY // LS is a new matrix ==================================================================

%% Baseline Comparison and export ---------------------------------------------
MSE_train_baseline = (reshape(Ytrain,[n,1]) - mean(mean(Ytrain)))'*(reshape(Ytrain, [n,1]) - mean(mean(Ytrain)))/n;
MSE_test_baseline = (reshape(Ytest,[n,1]) - mean(mean(Ytest)))'*(reshape(Ytest,[n,1]) - mean(mean(Ytest)))/n;
     
% Export experiment setup
% data for saving
data = [alpha beta MSE_train_baseline MSE_test_baseline];
dlmwrite('Data_setup_ER_50x100.csv', data, '-append');

% Baseline testing phase ------------------------------------------------------
gamma = sum(alpha_true_b);
Q = gamma * eye(n)+ beta_true_b*(LS);
  
% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha_true_b(j) * Rtest(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf;
end

% Export baseline
MSE_test_GCRF_BASE = (reshape(Ytest,[n,1]) - mu)' * (reshape(Ytest,[n,1]) - mu)/n;
R2 = 1 - MSE_test_GCRF_BASE/MSE_test_baseline;  

BaseResults = [alpha_true_b beta_true_b...
               MSE_train_GCRF_BASE MSE_test_GCRF_BASE R2 time_base iter_base];

%% Approximation testing phase --------------------------------------------------
gamma = sum(alpha_true_a);
Q = gamma*eye(n)+ beta_true_a*(LS);

% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha_true_a(j) * Rtest(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf; 
end

% Export approximation           
MSE_test_GCRF_IMPROVED = (reshape(Ytest,[n,1]) - mu)' * (reshape(Ytest,[n,1]) - mu)/n;
R2 = 1 - MSE_test_GCRF_IMPROVED/MSE_test_baseline;

ApproxResults = [alpha_true_a beta_true_a...
                 MSE_train_GCRF_IMPROVED MSE_test_GCRF_IMPROVED R2 time_improved iter_improved];
             
%% Approximation 2 testing phase -----------------------------------------------
gamma = sum(alpha_true_a2);
Q = gamma*eye(n)+ beta_true_a2*(LS);

% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha_true_a2(j) * Rtest(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf;
end

% Export approximation           
MSE_test_GCRF_IMPROVED2 = (reshape(Ytest,[n,1]) - mu)' * (reshape(Ytest,[n,1]) - mu)/n;
R2 = 1 - MSE_test_GCRF_IMPROVED2/MSE_test_baseline;

ApproxResults2 = [alpha_true_a2 beta_true_a2...
                 MSE_train_GCRF_IMPROVED2 MSE_test_GCRF_IMPROVED2 R2 time_improved2 iter_improved2];

%% Jesse testing phase --------------------------------------------------------
gamma = sum(alpha_true_j);
Q = gamma*eye(n)+ beta_true_j*(LS);

% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha_true_j(j) * Rtest(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);    
    mu = Q\bhalf; 
end

% Export MSN
MSE_test_GCRF_MSN = (reshape(Ytest,[n,1]) - mu)' * (reshape(Ytest,[n,1]) - mu)/n;
R2 = 1 - MSE_test_GCRF_MSN/MSE_test_baseline;

MSNResults = [alpha_true_j beta_true_j...
                MSE_train_GCRF_MSN MSE_test_GCRF_MSN R2 time_msn iter_msn];

end