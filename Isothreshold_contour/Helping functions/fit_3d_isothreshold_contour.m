function [fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, rgb_contour_cov,...
    ellParams] = fit_3d_isothreshold_contour(...
    rgb_ref, rgb_comp, grid_xyz, varargin)
    % Ensure the reference RGB has three components
    assert(length(rgb_ref) == 3);
    
    % Parse input arguments
    p = inputParser;
    p.addParameter('vecLength',[],@(x)(isnumeric(x)));
    p.addParameter('varyingRGBplane',[],@(x)(isnumeric(x)));
    p.addParameter('nThetaEllipse', 200, @isnumeric);
    p.addParameter('nPhiEllipse', 100, @isnumeric);
    p.addParameter('ellipse_scaler', 1, @isnumeric);

    % Extract parsed input arguments
    parse(p, varargin{:});
    nThetaEllipse   = p.Results.nThetaEllipse;
    nPhiEllipse     = p.Results.nPhiEllipse;
    ellipse_scaler  = p.Results.ellipse_scaler;
    vecLen          = p.Results.vecLength;
    idx_varyingDim  = p.Results.varyingRGBplane;
    
    % Generate a unit circle for ellipse fitting
    circleIn3D      = UnitCircleGenerate_3D(nThetaEllipse, nPhiEllipse);

    % Truncate the reference RGB to the specified dimensions
    rgb_ref_trunc   = rgb_ref(idx_varyingDim)';

    % Compute or use provided comparison stimuli
    if isempty(rgb_comp)
        % Compute the comparison stimuli if not provided
        rgb_comp_unscaled = reshape(rgb_ref_trunc,[1,1,3]) + ...
            repmat(vecLen,[1,1,3]).*grid_xyz;
        rgb_comp_scaled = reshape(rgb_ref_trunc,[1,1,3])+ ...
            repmat(vecLen,[1,1,3]).*ellipse_scaler.*grid_xyz;
    else
        % Use provided comparison stimuli
        rgb_comp_trunc = rgb_comp(idx_varyingDim,:)';
        rgb_comp_scaled = rgb_ref_trunc + rgb_comp_trunc.*ellipse_scaler;
    end

    % Compute covariance of the unscaled comparison stimuli
    rgb_comp_unscaled_reshape = reshape(rgb_comp_unscaled, [], 3);
    rgb_contour_cov = cov(rgb_comp_unscaled_reshape);

    %fit an ellipse
    [center, radii, evecs, v, chi2] = ellipsoid_fit(rgb_comp_unscaled_reshape, '');    

    ellParams = {center, radii, evecs, v, chi2};

    %Adjust the fitted ellipse based on the reference stimulus
    fitEllipse_unscaled = PointsOnEllipsoid(radii, center, evecs, circleIn3D);
    % Scale the fitted ellipse
    fitEllipse_scaled = (fitEllipse_unscaled - rgb_ref_trunc).*...
        ellipse_scaler + rgb_ref_trunc;
end