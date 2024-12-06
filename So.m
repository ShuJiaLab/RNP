function r = So(tau, X)                                                    
    % shrinkage operator
    r = sign(X) .* max(abs(X) - tau, 0);
end
