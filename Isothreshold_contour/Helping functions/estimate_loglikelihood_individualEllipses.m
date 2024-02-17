function nLogL = estimate_loglikelihood_individualEllipses(ellParam, ...
    x_ref, x_comp, alpha_Weibull, beta_Weibull, y)
    pInc  = predict_error_prob_individualEllipses(ellParam, x_ref,...
        x_comp, alpha_Weibull, beta_Weibull);
    pC    = 1 - pInc;
    logL  = y.*log(pC(:)+1e-20) + (1-y).*log(pInc(:)+1e-20);
    nLogL = -sum(logL(:));
end