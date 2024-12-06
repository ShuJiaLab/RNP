function [L, S] = RobustPCA(X, lambda, mu, tol, max_iter)                  
    % - X is a data matrix (of the size N x M) to be decomposed
    %   X can also contain NaN's for unobserved values
    % - lambda   - regularization parameter,           default = 1/sqrt(max(N,M))
    % - mu       - the augmented lagrangian parameter, default = 10*lambda
    % - tol      - reconstruction error tolerance,     default = 1e-6
    % - max_iter - maximum number of iterations,       default = 1000

    [M, N] = size(X);
    unobserved = isnan(X);
    X(unobserved) = 0;
    normX = norm(X, 'fro');

    % default arguments
    if nargin < 2
        lambda = 1 / sqrt(max(M,N));
    end
    if nargin < 3
        mu = 10*lambda;
    end
    if nargin < 4
        tol = 1e-6;
    end
    if nargin < 5
        max_iter = 1000;
    end
%     if nargin < 6
%         L0 = zeros(M, N);
%     end
    
    % initial solution
    L = zeros(M, N);
    S = zeros(M, N);
    Y = zeros(M, N);
    
    for iter = (1:max_iter)
        % ADMM step: update L and S
        L = Do(1/mu, X - S + (1/mu)*Y);
        S = So(lambda/mu, X - L + (1/mu)*Y);
        % and augmented lagrangian multiplier
        Z = X - L - S;
        Z(unobserved) = 0; % skip missing values
        Y = Y + mu*Z;
        
        err = norm(Z, 'fro') / normX;
        Err(iter) = err;

        if (iter == 1) || (mod(iter, 10) == 0) %|| (err < tol)
            fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
                    iter, err, rank(L), nnz(S(~unobserved)));
        end

        if (err <= tol && iter >= 500 && rank(L) <= 3) 
            break; 
        end
%         if iter > 1
%             if (err >= Err(iter-1))
%                 fprintf(1, 'iter: %04d\terr: %f\trank(L): %d\tcard(S): %d\n', ...
%                     iter, err, rank(L), nnz(S(~unobserved)));
%                 break; end
%         end
    end
end
