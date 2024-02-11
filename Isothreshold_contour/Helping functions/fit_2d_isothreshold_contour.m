function [fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, rgb_contour_cov,...
    ellParams, AConstraint, Ainv, Q, fitErr] = ...
    fit_2d_isothreshold_contour(rgb_ref, rgb_comp, ...
    grid_theta_xy, varargin)

    %throw an error is 
    assert(length(rgb_ref) == 3);
    
    p = inputParser;
    p.addParameter('vecLength',[],@(x)(isnumeric(x)));
    p.addParameter('varyingRGBplane',[],@(x)(isnumeric(x)));
    p.addParameter('nThetaEllipse', 200, @isnumeric);
    p.addParameter('ellipse_scaler', 1, @isnumeric);

    parse(p, varargin{:});
    nThetaEllipse   = p.Results.nThetaEllipse;
    ellipse_scaler  = p.Results.ellipse_scaler;
    vecLen          = p.Results.vecLength;
    idx_varyingDim  = p.Results.varyingRGBplane;
    
    %unit circle
    circleIn2D      = UnitCircleGenerate(nThetaEllipse);

    rgb_ref_trunc   = rgb_ref(idx_varyingDim)';

    if isempty(rgb_comp)
        % compute the iso-threshold contour 
        rgb_comp_unscaled = rgb_ref_trunc + repmat(vecLen,[1,2]).*grid_theta_xy';
        rgb_comp_scaled = rgb_ref_trunc + repmat(vecLen,[1,2]).*ellipse_scaler.*grid_theta_xy';
        % compute the covariance on the unscaled thresholds
        rgb_contour_cov = cov(rgb_comp_unscaled);
    
        %fit an ellipse
        [ellParams, AConstraint, Ainv, Q, fitErr] = FitEllipseQ(rgb_comp_unscaled' - ...
            rgb_ref_trunc','lockAngleAt0',false,'ratioMax',1000);
        fitEllipse_unscaled = (PointsOnEllipseQ(Q, circleIn2D) + rgb_ref_trunc')';
    
    else
        rgb_comp_trunc = rgb_comp(idx_varyingDim,:)';
        rgb_comp_scaled = rgb_ref_trunc + rgb_comp_trunc.*ellipse_scaler;
        rgb_contour_cov = cov(rgb_comp_trunc);
        %fit an ellipse
        [ellParams, AConstraint, Ainv, Q, fitErr] = FitEllipseQ(rgb_comp_trunc' - ...
            rgb_ref_trunc','lockAngleAt0',false,'ratioMax',1000);
        fitEllipse_unscaled = (PointsOnEllipseQ(Q, circleIn2D) + rgb_ref_trunc')';
    end
    %unscaled ellipse
    fitEllipse_scaled = (fitEllipse_unscaled - rgb_ref_trunc).*...
        ellipse_scaler + rgb_ref_trunc;
end