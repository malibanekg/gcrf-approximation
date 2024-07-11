addpath('GCRF_MSN - baseline');
addpath('GCRF_MSN - approx');
addpath('GCRF_MSN - approx2');
addpath('GCRF_MSN - proper_Jesse');
addpath('NearKronecker');
addpath('Data_generation');

NumIter = 30;

delete '*.csv';

%% with noise
% add header
fid = fopen('Data_setup_ER_30x50_shum.csv','W');
data = ['error1,','error2,','error3,', 'error4,', 'disconnected,', 'S1_del,','S1_grane,', 'S1_nule,','S2_del,', 'S2_grane,', 'S2_nule\n'];
fprintf(fid,data);
fclose(fid);

% add header baseline
fid = fopen('Experiments_ER_30x50_baseline_shum.csv','W');
% data = ['alpha_true_1,','alpha_true_2,','alpha_true_3,','beta_true,',...
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_BASE,', 'MSE_test_GCRF_BASE,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header baselineSVD
fid = fopen('Experiments_ER_30x50_baselineSVD_shum.csv','W');
% data = ['alpha_true_1,','alpha_true_2,','alpha_true_3,','beta_true,',...
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_BASESVD,', 'MSE_test_GCRF_BASESVD,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header approx
fid = fopen('Experiments_ER_30x50_approx_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED,', 'MSE_test_GCRF_IMPROVED,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header approx2
fid = fopen('Experiments_ER_30x50_approx2_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_IMPROVED2,', 'MSE_test_GCRF_IMPROVED2,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

% add header MSN
fid = fopen('Experiments_ER_30x50_msn_shum.csv','W');
data = ['alpha_true_1,','beta_true,',...
        'MSE_train_GCRF_MSN,', 'MSE_test_GCRF_MSN,', 'R2,', 'extime,', 'iterations\n'];
fprintf(fid,data);
fclose(fid);

%%
n1 = 30;
n2 = 50;
n = n1*n2;

tic
for i = 1:NumIter
    i
    % generate graphs for train                      % 43,  130,  217,  283,  348
    S1_main = GenRandGraphFixedNumLinksER(n1, 60);   % 122, 367,  612,  796,  980
    S2_main = GenRandGraphFixedNumLinksER(n2, 140);  % 495, 1485, 2475, 3217, 3960
	
	% 10%: 2(56) i 3(141)
	% 30%: 5(125) i 9(369)
	% 50%: 12(216) i 22(616)
	% 65%: 15(225) i 25(625)
	% S1_main = generate_random_graph(n1, 'ba', 3, -1); % 2, 5, 12, 15
	% S2_main = generate_random_graph(n2, 'ba', 4, -1); % 3, 9, 22, 25
	% 50x100
%     S1 = generate_random_graph(n1, 'ba', 4, -1); % 4, 9, 22, 25 
%     S2 = generate_random_graph(n2, 'ba', 8, -1); % 5, 18, 45, 50 
%     S1 = generate_random_graph(n1, 'ws', 6, -1);   % 5, 15, 25, 32, 40 
%     S2 = generate_random_graph(n2, 'ws', 10, -1); % 10, 30, 50, 65, 80
    Ytrain1 = normrnd(0,1,[n1,1]);
    Ytrain2 = normrnd(0,1,[n2,1]);
   
    Ytest1 = normrnd(0,1,[n1,1]);
    Ytest2 = normrnd(0,1,[n2,1]);
    
    % =========================================================
    % model training and testing phases
    % =========================================================
    [BaseResultss, BaseSVDResults, ApproxResultss, ApproxResults2s, MSNResultss] = DataSetup_shum(S1_main, S2_main, Ytrain1, Ytrain2, Ytest1, Ytest2);
    dlmwrite('Experiments_ER_30x50_baseline_shum.csv', BaseResultss, '-append');
    dlmwrite('Experiments_ER_30x50_baselineSVD_shum.csv', BaseSVDResults, '-append');
    dlmwrite('Experiments_ER_30x50_approx_shum.csv', ApproxResultss, '-append');
    dlmwrite('Experiments_ER_30x50_approx2_shum.csv', ApproxResults2s, '-append');
    dlmwrite('Experiments_ER_30x50_msn_shum.csv', MSNResultss, '-append');
end
toc
clear;
