function nLogL = estimate_loglikelihood_oddity(w_colvec, W_size, y, xref,...
    x0, x1, poly_chebyshev, varargin)

    p = inputParser;
    p.addParameter('etas', [], @(x)(isnumeric(x)));
    p.addParameter('num_MC_samples',100,@isnumeric);
    p.addParameter('scalePhi_toRGB',true,@islogical);

    parse(p, varargin{:});
    etas               = p.Results.etas;
    num_MC_samples     = p.Results.num_MC_samples;
    scalePhi_toRGB     = p.Results.scalePhi_toRGB;
   
    %W_size: [max_deg, max_deg, NUM_DIMS,NUM_DIMS+ EXTRA_DIMS]
    bandwidth = w_colvec(end);
    W     = reshape(w_colvec(1:end-1), W_size);
    pChoosingX1  = predict_error_prob_oddity(W, poly_chebyshev, xref, x0, x1,...
        'etas',etas,'num_MC_samples',num_MC_samples, 'scalePhi_toRGB',...
        scalePhi_toRGB, 'bandwidth', bandwidth);
    pChoosingX0    = 1 - pChoosingX1;
    logL  = (1-y).*log(pChoosingX1(:)+1e-20) + y.*log(pChoosingX0(:)+1e-20);
    nLogL = -sum(logL(:));
end