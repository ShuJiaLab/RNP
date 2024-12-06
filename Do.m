function r = Do(tau, X)                                                    
    % shrinkage operator for singular values
    [U, S, V] = svd(X, 'econ');
    r = U*So(tau, S)*V';
end
