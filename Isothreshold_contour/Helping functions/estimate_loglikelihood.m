function nLogL = estimate_loglikelihood(w_colvec, W_size, x, xbar, y, poly_chebyshev, varargin)

    p = inputParser;
    p.addParameter('etas', [], @(x)(isnumeric(x)));
    p.addParameter('num_MC_samples',100,@isnumeric);

    parse(p, varargin{:});
    etas               = p.Results.etas;
    num_MC_samples     = p.Results.num_MC_samples;
   
    %W_size: [max_deg, max_deg, NUM_DIMS,NUM_DIMS+ EXTRA_DIMS]
    W     = reshape(w_colvec, W_size);
    pInc  = predict_error_prob(W, poly_chebyshev, x, xbar, 'etas',etas,...
                'num_MC_samples',num_MC_samples);
    pC    = 1 - pInc;
    logL  = y.*log(pC(:)+1e-20) + (1-y).*log(pInc(:)+1e-20);
    nLogL = -sum(logL(:));
end