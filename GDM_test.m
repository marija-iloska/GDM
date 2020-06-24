clear all
clc


%% SETTINGS for generating data
dim_y = 8; var_u =1;
p_s = 0.7; p_ns = 0.3;
T = 1e3;

% Generating data and matrices
[A, C, y, dim_x] = generate_mat(T, dim_y, p_s, p_ns, var_u);



%% Bayesian Ridge Regression

%Consider a Gaussian prior
mu_0 = zeros(dim_x, 1);
var_0 = 1; sig_0 = var_0*eye(dim_x);

% Obtain the posterior of the vectorized matrix (and likelihood
% counterparts)
[mu_c, sig_c, mu_x, sig_x] = mn_conjugate_var(y, var_u, mu_0, sig_0);

% Convert to an estimate of the coefficient matrix
C_est = reshape(mu_c, dim_y, dim_y);

% Compute MSE in the coefficients
MSE = sum(sum((C-C_est).^2))/dim_x;


%% Gibbs Sampler

% Settings for Gibbs Bernoulli
I = 3000;                       % Gibbs iterations
I0 = 1500;                      % Gibbs burn-in 
K = 2;                          % Thinning parameter
A_init = ones(dim_y, dim_y);    % Initial adjacency matrix
R=2;
lambda_init = 5;                % Initial prior parameter
gamma = 2 : 1 : 8;                  % Hyperparameter for degree prior

tic
parfor run = 1:R

    [fs] = dm_f(A, I, I0, K, A_init, C_est, mu_x, sig_x, gamma, lambda_init)
    fs_dm(run, :) = fs;

end
toc

avg_dm = mean(fs_dm, 1);

save('DM.mat', 'avg_dm')
