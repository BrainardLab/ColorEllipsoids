function [fitEllipsoid_scaled, fitEllipsoid_unscaled, rgb_comp_scaled, ellParams] =...
    fit_3d_isothreshold_ellipsoid(rgb_ref, rgb_comp, grid_xyz, varargin)
    % Ensure the reference RGB has three components
    assert(length(rgb_ref) == 3);
    
    % Parse input arguments
    p = inputParser;
    p.addParameter('vecLength',[],@(x)(isnumeric(x)));
    p.addParameter('nThetaEllipsoid', 200, @isnumeric);
    p.addParameter('nPhiEllipsoid', 100, @isnumeric);
    p.addParameter('ellipsoid_scaler', 1, @isnumeric);

    % Extract parsed input arguments
    parse(p, varargin{:});
    nThetaEllipsoid   = p.Results.nThetaEllipsoid;
    nPhiEllipsoid     = p.Results.nPhiEllipsoid;
    ellipsoid_scaler  = p.Results.ellipsoid_scaler;
    vecLen            = p.Results.vecLength;
    
    % Generate a unit circle for Ellipsoid fitting
    circleIn3D      = UnitCircleGenerate_3D(nThetaEllipsoid, nPhiEllipsoid);

    % Compute or use provided comparison stimuli
    if isempty(rgb_comp)
        % Compute the comparison stimuli if not provided
        rgb_comp_unscaled = reshape(rgb_ref,[1,1,3]) + ...
            repmat(vecLen,[1,1,3]).*grid_xyz;
        rgb_comp_scaled = reshape(rgb_ref,[1,1,3])+ ...
            repmat(vecLen,[1,1,3]).*ellipsoid_scaler.*grid_xyz;
    else
        % Use provided comparison stimuli
        rgb_comp_unscaled = rgb_comp;
        rgb_comp_scaled = rgb_ref' + (rgb_comp_unscaled - rgb_ref').*ellipsoid_scaler;
    end

    % Compute covariance of the unscaled comparison stimuli
    if length(size(rgb_comp_scaled)) == 3
        rgb_comp_unscaled_reshape = reshape(rgb_comp_unscaled, [], 3);
    else
        rgb_comp_unscaled_reshape = rgb_comp_unscaled;
    end

    %fit an Ellipsoid
    [center, radii, evecs, v, chi2] = ellipsoid_fit(rgb_comp_unscaled_reshape, '');    

    ellParams = {center, radii, evecs, v, chi2};

    %Adjust the fitted Ellipsoid based on the reference stimulus
    fitEllipsoid_unscaled = PointsOnEllipsoid(radii, center, evecs, circleIn3D);
    % Scale the fitted Ellipsoid
    fitEllipsoid_scaled = (fitEllipsoid_unscaled - rgb_ref').*...
        ellipsoid_scaler + rgb_ref';
end