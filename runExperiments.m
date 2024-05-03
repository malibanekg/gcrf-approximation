addpath('GCRF_MSN - baseline');
addpath('GCRF_MSN - approx');
addpath('GCRF_MSN - approx2');
addpath('GCRF_MSN - proper_Jesse');
addpath('NearKronecker');
addpath('Data_generation');

NumIter = 30;

delete '*.csv';

%% Oficial documents, no noise
% add header
fid = fopen('Data_setup_ER_50x100.csv','W');
data = ['alpha01,','beta,',...
        'MSE_train_baseline,', 'MSE_test_baseline\n'];
fprintf(fid,data);
fclose(fid);

% add header baseline
fid = fopen('Experiments_ER_50x100_baseline.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_BASE,', 'MSE_test_GCRF_BASE,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header approx
fid = fopen('Experiments_ER_50x100_approx.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED,', 'MSE_test_GCRF_IMPROVED,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header approx2
fid = fopen('Experiments_ER_50x100_approx2.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED2,', 'MSE_test_GCRF_IMPROVED2,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header MSN
fid = fopen('Experiments_ER_50x100_msn.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_MSN,', 'MSE_test_GCRF_MSN,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);


% number of nodes for the graphs G and H
n1 = 50;
n2 = 100;

tic
for i = 1:NumIter
    % The additional commented numbers below represent the number of edges 
    % for the density levels of 10%, 30%, 50%, 65%, 80%
    
    % =========================================================
    % generate graphs for the training phase
    % =========================================================
    %% Erdos-Renyi networks -------------------------
    S1 = GenRandGraphFixedNumLinksER(n1,122); % 122, 367, 612, 796, 980
    S2 = GenRandGraphFixedNumLinksER(n2, 495); % 495, 1485, 2475, 3217, 3960 
    %% Barabasi-Albert networks ---------------------
    % S1 = generate_random_graph(n1, 'ba', 9, -1); % 4, 9, 22, 25 
    % S2 = generate_random_graph(n2, 'ba', 18, -1); % 5, 18, 45, 50
    %% Watts-Strogatz networks ----------------------
    % S1 = generate_random_graph(n1, 'ws', 40, -1);  % 5, 15, 25, 32, 40 
    % S2 = generate_random_graph(n2, 'ws', 80, -1); % 10, 30, 50, 65, 80
    
    Ytrain1 = normrnd(0,1,[n1,1]);
    Ytrain2 = normrnd(0,1,[n2,1]);
    
    % =========================================================
    % generate graphs for the testing phase
    % =========================================================
    %% Erdos-Renyi networks -------------------------
    S1t = GenRandGraphFixedNumLinksER(n1, 122);
    S2t = GenRandGraphFixedNumLinksER(n2, 495);
    %% Barabasi-Albert networks ---------------------
    % S1t = generate_random_graph(n1, 'ba', 9, -1);
    % S2t = generate_random_graph(n2, 'ba', 18, -1);
    %% Watts-Strogatz networks ----------------------
    % S1t = generate_random_graph(n1, 'ws', 40, -1);
    % S2t = generate_random_graph(n2, 'ws', 80, -1);
    
    Ytest1 = normrnd(0,1,[n1,1]);
    Ytest2 = normrnd(0,1,[n2,1]);

    % =========================================================
    % generate additional noise
    % =========================================================
    noise1 = normrnd(0,1/3,[n1,1]);
    noise2 = normrnd(0,1/3,[n2,1]);
    
    noise12 = normrnd(0,1/3,[n1,1]);
    noise22 = normrnd(0,1/3,[n2,1]);
    
    % =========================================================
    % model training and testing phases
    % =========================================================
    [BaseResults, ApproxResults, ApproxResults2, MSNResults] = DataSetup(S1, S2, Ytrain, Rtrain, S1t, S2t, Ytest, Rtest, noise1, noise2, noise12, noise22); 
    dlmwrite('Experiments_ER_50x100_baseline.csv', BaseResults, '-append');
    dlmwrite('Experiments_ER_50x100_approx.csv', ApproxResults, '-append');
    dlmwrite('Experiments_ER_50x100_approx2.csv', ApproxResults2, '-append');
    dlmwrite('Experiments_ER_50x100_msn.csv', MSNResults, '-append');
end
toc
clear;
