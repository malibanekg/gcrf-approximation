function [BaseResults, BaseSVDResults, ApproxResults, ApproxResults2, MSNResults] = DataSetup_shum(S1_main, S2_main, Ytrain1, Ytrain2, Ytest1, Ytest2)

%% TRAIN SETTINGS
% rng(1000);
format long;

[n1,~] = size(S1_main);
[n2,~] = size(S2_main);
n = n1 * n2;
disconnected = 0;
% Sim = kron(S1,S2);
% tt = rank(Sim);
% bla = Sim;

%% Structure of the initial matrix is unknown
% Approximated initial matrices with Kronecker product of graphs
% number of the blocks
m1 = 30;
n1 = 30;

% size of the blocks
% Treba da se dobije velicina bloka kao matrica S2
m2 = n/m1;
n2 = n/n1;

%% add noise ------------------------
% rng('shuffle');
Ytrain = kron(Ytrain1, Ytrain2);
% Ytrain_original = Ytrain;

% Ytrain_noise = Ytrain; % + normrnd(0,0.33,[n,1]);

%% Similarity
Ytrain1_noise = Ytrain1 + normrnd(0,0.25,[n1,1]);
Ytrain2_noise = Ytrain2 + normrnd(0,0.25,[n2,1]);
S1 = zeros(n1,n1);
S2 = zeros(n2, n2);

for i = 1:n1-1
    for j = (i+1):n1
        if S1_main(i,j) == 1
            S1(i,j) = exp(-abs(Ytrain1_noise(i) - Ytrain1_noise(j))); % + normrnd(0,0.1);
            S1(j,i) = S1(i,j);
        else
            S1(i,j) = 0;
            S1(j,i) = S1(i,j);
        end
    end
end

for i = 1:n2-1
    for j = (i+1):n2
        if S2_main(i,j) == 1
            S2(i,j) = exp(-abs(Ytrain2_noise(i) - Ytrain2_noise(j))); % + normrnd(0,0.1);
            S2(j,i) = S2(i,j);
        else
            S2(i,j) = 0;
            S2(j,i) = S2(i,j);
        end
    end
end

% shit
% s1 = norm(S1,'fro');
% S1 = S1/s1;
% s2 = norm(S1,'fro');
% S2 = S2/s2;
% shit

Sim = kron(S1, S2);

error1 = norm(Sim - kron(S1, S2),'fro');

generateEdges(1500, 1680);  %% 30%: 4771, 9542; 14313, 19084; 30%=28626; 40%=38168
                            % 10%: 524,  1049; 1572, 2098;  30%= 3147; 40%=4196;  8350 - 80%
							% 10%_2nd: 840, 1680, 2520, 3360
                            % 10%30%: 2202, 4404; 6606, 8808; 40%=17616; 60%=26424
                            % 10%20%: 1470, 2940; 4410, 5880; 40%=11760; 60%=17640
                            % 20%10%: 1586, 3172; 4758, 6344; 40%=12688; 60% = 19032
                            % 30%10%: 2647, 5294; 7942, 10589; 40%=21176; 60%=31764 
                            % 20%20%: 2131, 4263; 6394, 8526; 40%=17052; 60%=25578
							% 50%50%: 13280, 26560, 
							% 100x200 == 500, 2000
							% 5%: 25000, 10%: 50000
matrica = load('RandomEdges.mat','-mat');
Sim_positions = matrica.Sim_positions;
clear matrica;

%% Dodavanje shuma - noise
for position_x = 1:(n - 1) % (m1 * m2)  
    for position_y = (position_x + 1):n
        if Sim_positions(position_x, position_y) == 1
            % graph S1
            u1 = floor((position_x - 1)/50) + 1;
            u2 = floor((position_y - 1)/50) + 1;
            % graph S2
            v1 = position_x - floor((position_x - 1)/50)*50;
            v2 = position_y - floor((position_y - 1)/50)*50;

            Sim(position_x, position_y) = (exp(-abs(Ytrain1_noise(u1) - Ytrain1_noise(u2))) * exp(-abs(Ytrain2_noise(v1) - Ytrain2_noise(v2)))); %/(s1 * s2);
            Sim(position_y, position_x) = Sim(position_x, position_y);
        else
            
        end        
    end
end
clear Sim_positions;
% clear diagonal - just in case
Sim = Sim - diag(diag(Sim));

error2= norm(Sim - kron(S1,S2),'fro');

%% SVD approximacija i dobijanje matrica S1' i S2' ==> Pocetni grafovi S1 i S2
tic
P = permMatrix(Sim, m1, n1, m2, n2);
matrix_rank = 1;
[U,Sing,V] = svds(P,matrix_rank);
Sigma = diag(Sing);
% ----------- rank = 1
vecB = U(:, 1); % sqrt(Sigma(1)) * 
vecC = Sigma(1) * V(:, 1);

S1prima = abs(vec2block(vecB, m1, n1));
S2prima = abs(vec2block(vecC, m2, n2));
S1pom = S1prima;
S2pom = S2prima;
% % numeric problem with zeros:
% indices = (S1prima < 1e-06);  
% S1prima(indices) = 0;
% % S1prima = S1prima/norm(S1prima, 'fro');
% indices = (S2prima < 1e-06);  
% S2prima(indices) = 0;
% % S2prima = S2prima/norm(S2prima, 'fro');

% proredjivanje grafova -------------------------
vektorS = reshape(S1prima, [900,1]); % 900
quant = quantile(vektorS,[.5 .88]);
quant(2)
indices = (S1prima < quant(2)); % 
S1prima(indices) = 0;
S1prima = S1prima - diag(diag(S1prima));
% strcat({'ima grana: '},  num2str(sum(sum(S1prima ~= 0))/2), {' nula: '}, num2str(sum(sum(S1prima == 0))/2))

vektorS = reshape(S2prima, [2500,1]); % 2500
quant = quantile(vektorS,[.5 .88]);
quant(2)
indices = (S2prima < quant(2));
S2prima(indices) = 0;
S2prima = S2prima - diag(diag(S2prima));
% strcat({'ima grana: '},  num2str(sum(sum(S2prima ~= 0))/2), {' nula: '}, num2str(sum(sum(S2prima == 0))/2))

error3 = norm(Sim - kron(S1prima,S2prima),'fro');

% % S1 ----------------------------------------------------------------------
% % lim1 = mean(mean(S1prima)); % median(median(S1prim));
% vektorS = S1prima(triu(S1prima)~=0);
% quant = quantile(vektorS,[.3 .25 .5 .75]);
% upper = quant(4) + 3 * (quant(4) - quant(2));
% lower = quant(2) - 3 * (quant(4) - quant(2));
% 
% lim1 = quant(1);
% 
% strcat(num2str(lower), {' - '}, num2str(quant(1)), {' - '}, num2str(quant(3)), {' - '}, num2str(upper))
% 
% how_edges = sum(S1prima ~= 0, 2); % koliko ima grana
% del_edges = sum((S1prima <= lim1 & S1prima ~= 0), 2); % koliko bi grana trebalo obrisati (!=0 and < lim1);;; (S1prima <= lim1 & S1prima ~= 0)
% indices = ((S1prima <= lim1 & S1prima ~= 0)) & repelem(how_edges > del_edges, 1, n1); % |(S1prima > upper)
% strcat({'ostalo grana: '},  num2str((sum(sum(S1prima ~= 0)) - sum(sum(indices)))/2), {' obrisano: '},  num2str(sum(sum(indices))), {' nula: '}, num2str((sum(sum(S1prima == 0)) + sum(sum(indices)))/2))                                
% S1prima(indices) = 0;
% 
% if (sum(how_edges == del_edges) > 0)
%     disconnected = 1;
% %     maxs = repelem(max(S1prima, [], 2),1, n1);
% %     indices = repelem(how_edges == del_edges, 1, n1); % & (S1prima ~= maxs);
% %     S1prima(indices) = 1;
% end
% 
% S1_del = sum(sum(indices))/2;
S1_del = 0;
% 
% % S2 ----------------------------------------------------------------------
% % lim2 = mean(mean(S2prima));
% vektorS = S2prima(triu(S2prima)~=0);
% quant = quantile(vektorS,[.3 .25 .5 .75]);
% upper = quant(4) + 3 * (quant(4) - quant(2));
% lower = quant(2) - 3 * (quant(4) - quant(2));
% 
% lim2 = quant(1);
% 
% strcat(num2str(lower), {' - '}, num2str(quant(1)), {' - '}, num2str(quant(3)), {' - '}, num2str(upper))
% 
% how_edges = sum(S2prima ~= 0, 2); % koliko ima grana
% del_edges = sum((S2prima <= lim2 & S2prima ~= 0), 2); % koliko bi grana trebalo obrisati;;;; (S2prima <= lim2 & S2prima ~= 0)
% indices = ((S2prima <= lim2 & S2prima ~= 0)) & repelem(how_edges > del_edges, 1, n2); % |(S2prima > upper)
% strcat({'ostalo grana: '},  num2str((sum(sum(S2prima ~= 0)) - sum(sum(indices)))/2), {' obrisano: '},  num2str(sum(sum(indices))), {' nula: '}, num2str((sum(sum(S2prima == 0)) + sum(sum(indices)))/2))
% S2prima(indices) = 0;
% 
% if (sum(how_edges == del_edges) > 0)
%     disconnected = 1;
% %     maxs = repelem(min(S2prima, [], 2),1, n2);
% %     indices = repelem(how_edges == del_edges, 1, n2); % & (S2prima ~= maxs);
% %     S2prima(indices) = 1;
% end
 
% S2_del = sum(sum(indices))/2;
S2_del = 0;

error4 = norm(Sim - kron(S1prima,S2prima),'fro');

%% Export experiment setup
% data = [tt rank(Sim) rank(P) error2];
% data = [error1 error2 error3 error4 disconnected];
% dlmwrite('Data_setup_ER_30x50_shum.csv', data, '-append');

S1g = sum(sum(S1prima ~= 0))/2;
S1n = sum(sum(S1prima == 0))/2;
S2g = sum(sum(S2prima ~= 0))/2;
S2n = sum(sum(S2prima == 0))/2;

data = [error1 error2 error3 error4 disconnected S1_del S1g S1n S2_del S2g S2n];
dlmwrite('Data_setup_ER_30x50_shum.csv', data, '-append');

%% GCRF train setup; Generate parameters Ytrain and Rtrain
% Ytrain = kron(Ytrain1, Ytrain2);
t = 1;
beta = 5; 
alpha = 1;
k = numel(alpha);
gamma = sum(alpha);

LS = diag(sum(Sim)) - Sim;
Q = gamma*eye(n)+ beta*(LS);
clearvars LS;
% shum za Ytrain
rob = normrnd(0,0.33,[n, t]);
Ytraintr = Ytrain + rob(:,t);

bhalf = Q * Ytraintr;
Rtrain = bhalf;

Ytrain = reshape(Ytrain,[n2,n1]);
Rtrain = reshape(Rtrain,[n2,n1]);
 
%% TRAINING PHASE 
% Baseline model
tic
%[alpha_r,beta_r,MSE_train_GCRF_BASE, output] = MSN_train_base(Ytrain,Rtrain,S1prim,S2prim);
[alpha_r,beta_r,MSE_train_GCRF_BASE, output] = MSN_train_base_shum(Ytrain,Rtrain,Sim, n1, n2);
time_base = toc;
iter_base = output.iterations;
% alpha_true_b = alpha_r/sum(alpha_r);
% beta_true_b = beta_r/sum(alpha_r);
alpha_true_b = alpha_r;
beta_true_b = beta_r;

%%
% clear S1 S2;
% S1prima = S1;
% S2prima = S2;
%% Baseline model SVD
tic
[alpha_r,beta_r,MSE_train_GCRF_BASESVD, output] = MSN_train_base(Ytrain,Rtrain,S1pom,S2pom);
time_basesvd = toc;
iter_basesvd = output.iterations;
% alpha_true_b = alpha_r/sum(alpha_r);
% beta_true_b = beta_r/sum(alpha_r);
alpha_true_bsvd = alpha_r;
beta_true_bsvd = beta_r;

% Approximated model 
tic
[alpha_r,beta_r,MSE_train_GCRF_IMPROVED, output] = MSN_train_approx(Ytrain,Rtrain,S1prima,S2prima);
time_improved = toc;
iter_improved = output.iterations;
% alpha_true_a = alpha_r/sum(alpha_r);
% beta_true_a = beta_r/sum(alpha_r);
alpha_true_a = alpha_r;
beta_true_a = beta_r;

% Approximated model 2
tic
[alpha_r,beta_r,MSE_train_GCRF_IMPROVED2, output] = MSN_train_approx2(Ytrain,Rtrain,S1prima,S2prima);
time_improved2 = toc;
iter_improved2 = output.iterations;
% alpha_true_a2 = alpha_r/sum(alpha_r);
% beta_true_a2 = beta_r/sum(alpha_r);
alpha_true_a2 = alpha_r;
beta_true_a2 = beta_r;

% Jesse model
tic
[alpha_r,beta_r, MSE_train_GCRF_MSN, output] = MSN_train_jesse(Ytrain,Rtrain,S1prima,S2prima); 
time_msn = toc;
iter_msn = output.iterations;
% alpha_true_j = alpha_r/sum(alpha_r);
% beta_true_j = beta_r/sum(alpha_r);
alpha_true_j = alpha_r;
beta_true_j = beta_r;

%% =======================================================================
% TEST SETTINGS ==========================================================
% ========================================================================

Ytest = kron(Ytest1, Ytest2);

% % Similarity for S1 i S2
% Ytest1_noise = Ytest1 + normrnd(0,0.33,[n1,1]);
% Ytest2_noise = Ytest2 + normrnd(0,0.33,[n2,1]);
% S1 = zeros(n1,n1);
% S2 = zeros(n2,n2);
% 
% for i = 1:n1-1
%     for j = (i+1):n1
%         if S1_main(i,j) == 1
%             S1(i,j) = exp(-abs(Ytest1_noise(i) - Ytest1_noise(j)));
%             S1(j,i) = S1(i,j);
%         else
%             S1(i,j) = 0;
%             S1(j,i) = S1(i,j);
%         end
%     end
% end
% 
% for i = 1:n2-1
%     for j = (i+1):n2
%         if S2_main(i,j) == 1
%             S2(i,j) = exp(-abs(Ytest2_noise(i) - Ytest2_noise(j)));
%             S2(j,i) = S2(i,j);
%         else
%             S2(i,j) = 0;
%             S2(j,i) = S2(i,j);
%         end
%     end
% end
% Sim_t = kron(S1,S2);
%  
% for position_x = 1:(n - 1) % (m1 * m2)  
%     for position_y = (position_x + 1):n
%         if Sim_positions(position_x, position_y) == 1
%             % graph S1
%             u1 = floor((position_x - 1)/50) + 1;
%             u2 = floor((position_y - 1)/50) + 1;
%             % graph S2
%             v1 = position_x - floor((position_x - 1)/50)*50;
%             v2 = position_y - floor((position_y - 1)/50)*50;
% 
%             Sim_t(position_x, position_y) = exp(-abs(Ytest1_noise(u1) - Ytest1_noise(u2))) * exp(-abs(Ytest2_noise(v1) - Ytest2_noise(v2)));
%             Sim_t(position_y, position_x) = Sim_t(position_x, position_y);           
%         end        
%     end
% end

%% SVD for test
% disconnected2 = 0;
% P = permMatrix(Sim_t, m1, n1, m2, n2);
% matrix_rank = 1;
% [U,Sing,V] = svds(P,matrix_rank);
% Sigma = diag(Sing);
% % ----------- rank = 1
% vecB = sqrt(Sigma(1)) * U(:, 1);
% vecC = sqrt(Sigma(1)) * V(:, 1);
% S1tprim = abs(vec2block(vecB, m1, n1));
% S2tprim = abs(vec2block(vecC, m2, n2));
% 
% % -------------------------------------------
% % to 0/1
% % S1
% lim1 = mean(mean(S1tprim)); % ;
% how_edges = sum(S1tprim ~= 0, 2); % koliko ima grana
% del_edges = sum(S1tprim <= lim1 & S1tprim ~= 0, 2); % koliko bi grana trebalo obrisati (!=0 and < lim1)
% indices = S1tprim <= lim1 & repelem(how_edges > del_edges, 1, n1);
% S1tprim(indices) = 0;
% 
% if (sum(how_edges == del_edges) > 0)
%     disconnected2 = 1;
% %   maxs = repelem(max(S1prima, [], 2),1, n1);
% %     indices = repelem(how_edges == del_edges, 1, n1); % & (S1prima ~= maxs);
% %     S1tprim(indices) = 1;
% end
% S1tprim = S1tprim - diag(diag(S1tprim));
% 
% % S2
% lim2 = mean(mean(S2tprim));
% how_edges = sum(S2tprim ~= 0, 2); % koliko ima grana
% del_edges = sum(S2tprim <= lim2 & S2tprim ~= 0, 2); % koliko bi grana trebalo obrisati (!=0 and < lim1)
% indices = S2tprim <= lim2 & repelem(how_edges > del_edges, 1, n2);
% S2tprim(indices) = 0;
% 
% if (sum(how_edges == del_edges) > 0)
%     disconnected2 = 1;
% %     maxs = repelem(max(S1prima, [], 2),1, n1);
% %     indices = repelem(how_edges == del_edges, 1, n2); % & (S1prima ~= maxs);
% %     S2tprim(indices) = 1;
% end
% S2tprim = S2tprim - diag(diag(S2tprim));

%% Test parameters -----
LS = diag(sum(Sim)) - Sim;
clearvars Sim;
Q = gamma*eye(n)+ beta*(LS);
% shum za Ytrain
rob = normrnd(0,0.33,[n, t]);
Ytesttr = Ytest + rob(:,t);

bhalf = Q * Ytesttr;
Rtest = bhalf;
Ytest = reshape(Ytest,[n2,n1]);
Rtest = reshape(Rtest,[n2,n1]);

MSE_test_baseline = (reshape(Ytest,[n,1]) - mean(mean(Ytest)))'*(reshape(Ytest,[n,1]) - mean(mean(Ytest)))/n;
     
%% Baseline testing phase
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

%% Novo LS
clearvars LS;
Sprim = kron(S1prima,S2prima);
% Sprim = kron(S1tprim,S2tprim);
LS = diag(sum(Sprim)) - Sprim;
clearvars Sprim;
clearvars S1prima S2prima S1tprim S2tprim;

%% BaselineSVD testing phase
gamma = sum(alpha_true_bsvd);
Q = gamma * eye(n)+ beta_true_bsvd*(LS);
  
% calculate error
for i = 1:t
    bhalf=0;
    for j=1:k
        bhalf = bhalf + alpha_true_bsvd(j) * Rtest(:,:,i,j);
    end
    bhalf = reshape(bhalf,[n,1]);
    mu = Q\bhalf;
end

% Export baseline
MSE_test_GCRF_BASESVD = (reshape(Ytest,[n,1]) - mu)' * (reshape(Ytest,[n,1]) - mu)/n;
R2 = 1 - MSE_test_GCRF_BASESVD/MSE_test_baseline;  

BaseSVDResults = [alpha_true_bsvd beta_true_bsvd...
               MSE_train_GCRF_BASESVD MSE_test_GCRF_BASESVD R2 time_basesvd iter_basesvd];
           
%% Approximation 2 testing phase
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
           
%% Approximation testing phase
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
             
%% Jesse testing phase
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