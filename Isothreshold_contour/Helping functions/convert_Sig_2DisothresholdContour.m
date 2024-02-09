function [recover_fitEllipse, recover_fitEllipse_unscaled, ...
    recover_rgb_contour, recover_rgb_contour_cov, recover_rgb_comp_est] = ...
    convert_Sig_2DisothresholdContour(rgb_ref, varying_RGBplane, ...
    grid_theta_xy, vecLength, pC_threshold, coeffs_chebyshev, W, varargin)

    p = inputParser;
    p.addParameter('nThetaEllipse', 200, @isnumeric);
    p.addParameter('contour_scaler',5, @isnumeric);
    p.addParameter('nSteps_bruteforce',1000, @isnumeric);

    parse(p, varargin{:});
    nThetaEllipse      = p.Results.nThetaEllipse;
    contour_scaler     = p.Results.contour_scaler;
    nSteps_bruteforce  = p.Results.nSteps_bruteforce;

    %initialize 
    numDirPts = size(grid_theta_xy,2);
    recover_vecLength = NaN(1,numDirPts);
    recover_rgb_comp_est = NaN(numDirPts, 3);

    %grab the reference stimulus's RGB
    rgb_ref_t = rgb_ref(1,1,varying_RGBplane); %truncated
    rgb_ref_s = squeeze(rgb_ref_t); %squeezed

    %for each chromatic direction
    for k = 1:numDirPts
        %determine the direction we are going 
        vecDir = grid_theta_xy(:,k);
        
        %run fmincon to search for the magnitude of vector that
        %leads to a pre-determined deltaE
        for l = 1:nSteps_bruteforce
            rgb_comp = rgb_ref_t;
            rgb_comp(1,1,:) = squeeze(rgb_comp(1,1,:))+ vecDir.*vecLength(l);
            pInc(l) = predict_error_prob(W, coeffs_chebyshev,...
                rgb_comp, rgb_ref_t);
        end
        [~, min_idx] = min(abs((1-pInc) - pC_threshold));
        recover_vecLength(k) = vecLength(min_idx);
        recover_rgb_comp_est(k,varying_RGBplane) = rgb_ref_s + ...
            contour_scaler.*vecDir.*vecLength(min_idx);
    end

    fixed_RGBplane = setdiff(1:3,varying_RGBplane);
    recover_rgb_comp_est(:,fixed_RGBplane) = rgb_ref(1,1,fixed_RGBplane);

    [recover_fitEllipse, recover_fitEllipse_unscaled, ...
        recover_rgb_contour, recover_rgb_contour_cov, ~,~,~,~] = ...
        fit_2d_isothreshold_contour(squeeze(rgb_ref), recover_vecLength',...
        varying_RGBplane, grid_theta_xy,...
        'nThetaEllipse',nThetaEllipse,...
        'ellipse_scaler',contour_scaler);
end
