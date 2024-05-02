addpath('GCRF_MSN - baseline');
addpath('GCRF_MSN - approx');
addpath('GCRF_MSN - approx2');
addpath('GCRF_MSN - proper_Jesse');
addpath('NearKronecker');
addpath('Data_generation');

NumIter = 1;

delete '*.csv';

%% Oficial documents, no noise
% add header
fid = fopen('Data_setup_ER_50x100.csv','W');
% data = ['alpha01,','alpha02,','alpha03,','beta,',...
data = ['alpha01,','beta,',...
        'MSE_train_baseline,', 'MSE_test_baseline\n'];
fprintf(fid,data);
fclose(fid);
% add header baseline
fid = fopen('Experiments_ER_50x100_baseline.csv','W');
% data = ['alpha_true_1,','alpha_true_2,','alpha_true_3,','beta_true,',...
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

%% with noise
% add header
fid = fopen('Data_setup_ER_50x100_shum.csv','W');
% data = ['alpha01,','alpha02,','alpha03,','beta,',...
data = ['original_rank,','sim_rank,','perm_rank,', 'error\n'];
fprintf(fid,data);
fclose(fid);
% add header baseline
fid = fopen('Experiments_ER_50x100_baseline_shum.csv','W');
% data = ['alpha_true_1,','alpha_true_2,','alpha_true_3,','beta_true,',...
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_BASE,', 'MSE_test_GCRF_BASE,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);
% add header baselineSVD
fid = fopen('Experiments_ER_50x100_baselineSVD_shum.csv','W');
% data = ['alpha_true_1,','alpha_true_2,','alpha_true_3,','beta_true,',...
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_BASESVD,', 'MSE_test_GCRF_BASESVD,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);
% add header approx
fid = fopen('Experiments_ER_50x100_approx_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED,', 'MSE_test_GCRF_IMPROVED,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);
% add header approx2
fid = fopen('Experiments_ER_50x100_approx2_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED2,', 'MSE_test_GCRF_IMPROVED2,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);
% add header MSN
fid = fopen('Experiments_ER_50x100_msn_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_MSN,', 'MSE_test_GCRF_MSN,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

%%
n1 = 50;
n2 = 100;

tic
for i = 1:NumIter
    i
    % generate graphs for train
    S1 = GenRandGraphFixedNumLinksER(n1,122);   % 122, 367, 612, 796, 980
    S2 = GenRandGraphFixedNumLinksER(n2, 495); % 495, 1485, 2475, 3217, 3960
%     S1 = generate_random_graph(n1, 'ba', 9, -1); % 4, 9, 22, 25 
%     S2 = generate_random_graph(n2, 'ba', 18, -1); % 5, 18, 45, 50 
%     S1 = generate_random_graph(n1, 'ws', 40, -1);   % 5, 15, 25, 32, 40 
%     S2 = generate_random_graph(n2, 'ws', 80, -1); % 10, 30, 50, 65, 80
    Ytrain1 = normrnd(0,1,[n1,1]);
    Ytrain2 = normrnd(0,1,[n2,1]);
    % generate graphs for test
    S1t = GenRandGraphFixedNumLinksER(n1, 122);
    S2t = GenRandGraphFixedNumLinksER(n2, 495);
%     S1t = generate_random_graph(n1, 'ba', 9, -1);
%     S2t = generate_random_graph(n2, 'ba', 18, -1);
%     S1t = generate_random_graph(n1, 'ws', 40, -1);
%     S2t = generate_random_graph(n2, 'ws', 80, -1);

    Ytest1 = normrnd(0,1,[n1,1]);
    Ytest2 = normrnd(0,1,[n2,1]);
%     % noises
%     noise1 = normrnd(0,1/3,[n1,1]);
%     noise2 = normrnd(0,1/3,[n2,1]);
%     
%     noise12 = normrnd(0,1/3,[n1,1]);
%     noise22 = normrnd(0,1/3,[n2,1]);
    
    % train and test with noise
    [BaseResultss, BaseSVDResults, ApproxResultss, ApproxResults2s, MSNResultss, Ytrain, Rtrain, Ytest, Rtest] = DataSetup_shum(S1, S2, Ytrain1, Ytrain2, S1t, S2t, Ytest1, Ytest2);
    %  , noise1, noise2, noise12, noise22
    dlmwrite('Experiments_ER_50x100_baseline_shum.csv', BaseResultss, '-append');
    dlmwrite('Experiments_ER_50x100_baselineSVD_shum.csv', BaseSVDResults, '-append');
    dlmwrite('Experiments_ER_50x100_approx_shum.csv', ApproxResultss, '-append');
    dlmwrite('Experiments_ER_50x100_approx2_shum.csv', ApproxResults2s, '-append');
    dlmwrite('Experiments_ER_50x100_msn_shum.csv', MSNResultss, '-append');
    % train and test without noise  // 
%     [BaseResults, ApproxResults, ApproxResults2, MSNResults] = DataSetup(S1, S2, Ytrain, Rtrain, S1t, S2t, Ytest, Rtest, noise1, noise2, noise12, noise22); 
%     dlmwrite('Experiments_ER_50x100_baseline.csv', BaseResults, '-append');
%     dlmwrite('Experiments_ER_50x100_approx.csv', ApproxResults, '-append');
%     dlmwrite('Experiments_ER_50x100_approx2.csv', ApproxResults2, '-append');
%     dlmwrite('Experiments_ER_50x100_msn.csv', MSNResults, '-append');
end
toc
clear;
