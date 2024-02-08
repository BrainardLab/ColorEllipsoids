function [fitEllipse_scaled, fitEllipse_unscaled, rgb_contour, ...
    rgb_contour_cov, ellParams, AConstraint, Ainv, Q, fitErr] = ...
    fit_2d_isothreshold_contour(rgb_ref, vecLen, idx_varyingDim, ...
    grid_theta_xy, varargin)

    %throw an error is 
    assert(length(rgb_ref) == 3);
    assert(length(vecLen) == size(grid_theta_xy,2));
    assert(size(grid_theta_xy,1)==length(idx_varyingDim));
    
    p = inputParser;
    p.addParameter('nThetaEllipse', 200, @isnumeric);
    p.addParameter('ellipse_scaler', 1, @isnumeric);

    parse(p, varargin{:});
    nThetaEllipse = p.Results.nThetaEllipse;
    ellipse_scaler      = p.Results.ellipse_scaler;
    
    %unit circle
    circleIn2D      = UnitCircleGenerate(nThetaEllipse);

    rgb_ref_trunc   = rgb_ref(idx_varyingDim)';
    % compute the iso-threshold contour 
    rgb_contour     = rgb_ref_trunc + repmat(vecLen.*ellipse_scaler, [1,2]).*grid_theta_xy';
    rgb_contour_cov = cov(rgb_contour);

    %fit an ellipse
    [ellParams, AConstraint, Ainv, Q, fitErr] = FitEllipseQ(rgb_contour' - ...
        rgb_ref_trunc','lockAngleAt0',false);
    fitEllipse_scaled = (PointsOnEllipseQ(Q, circleIn2D) + rgb_ref_trunc')';

    %unscaled ellipse
    fitEllipse_unscaled = (fitEllipse_scaled - rgb_ref_trunc)./...
        ellipse_scaler + rgb_ref_trunc;
end