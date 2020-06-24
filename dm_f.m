function [fs] = dm_f(A, I, I0, K, A_init, C_est, mu_x, sig_x, gamma, lambda_init)

for prior = 1:length(gamma)
    
    % Call Gibbs
    tic
    [A_s] = gibbs_1(I, I0, K, A_init, lambda_init, C_est, mu_x, sig_x, gamma(prior));
    toc 
    
    % Estimate as the mode
    A_est = mode(A_s, 3);

    % Performance
    [~, ~, fs] = adj_eval(A, A_est);
    fs_3(prior) = fs;      
    
end


end