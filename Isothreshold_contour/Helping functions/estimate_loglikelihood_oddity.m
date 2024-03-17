function logL = estimate_loglikelihood_oddity(W, model, y, xref, x0, x1,...
    num_samples, bandwidth)
    % compute (Uref, U0, U1) to specify ellipsoids
    Uref = compute_U(poly_chebyshev, W, xref);
end