function [fitEllipse_scaled, fitEllipse_unscaled, rgb_comp_scaled, rgb_contour_cov,...
    ellParams, AConstraint, Ainv, Q, fitErr] = fit_2d_isothreshold_contour(...
    rgb_ref, rgb_comp, grid_theta_xy, varargin)
% Fits an ellipse to an isothreshold contour in 2D space. The contour can be
% specified directly through comparison stimuli or computed from vector lengths
% along specified chromatic directions. The function supports scaling the fitted
% ellipse.
%
% Inputs:
%   rgb_ref - The RGB values of the reference stimulus.
%   rgb_comp - The RGB values of the comparison stimuli. If empty, the comparison
%              stimuli are computed using vector lengths along specified directions.
%   grid_theta_xy - Unit vectors along specified chromatic directions.
%   varargin - Additional optional parameters including:
%       'vecLength' - Vector lengths for computing comparison stimuli if not directly provided.
%       'varyingRGBplane' - Indices of the RGB dimensions to consider.
%       'nThetaEllipse' - Number of points to generate for the unit circle used in fitting.
%       'ellipse_scaler' - Scaling factor for the ellipse.
%
% Outputs:
%   fitEllipse_scaled - The scaled fitted ellipse centered around the reference stimulus.
%   fitEllipse_unscaled - The original fitted ellipse before scaling.
%   rgb_comp_scaled - Scaled RGB values of the comparison stimuli.
%   rgb_contour_cov - Covariance of the RGB values of the comparison stimuli.
%   ellParams - Parameters of the fitted ellipse.
%   AConstraint - The constraint matrix for the ellipse fitting.
%   Ainv - The inverse of the constraint matrix.
%   Q - The quadratic form of the fitted ellipse.
%   fitErr - The fitting error.

    % Ensure the reference RGB has three components
    assert(length(rgb_ref) == 3);
    
    % Parse input arguments
    p = inputParser;
    p.addParameter('vecLength',[],@(x)(isnumeric(x)));
    p.addParameter('varyingRGBplane',[],@(x)(isnumeric(x)));
    p.addParameter('nThetaEllipse', 200, @isnumeric);
    p.addParameter('ellipse_scaler', 1, @isnumeric);
    p.addParameter('nRepeats', 5, @isnumberic);

    % Extract parsed input arguments
    parse(p, varargin{:});
    nThetaEllipse   = p.Results.nThetaEllipse;
    ellipse_scaler  = p.Results.ellipse_scaler;
    vecLen          = p.Results.vecLength;
    idx_varyingDim  = p.Results.varyingRGBplane;
    nRepeats        = p.Results.nRepeats;
    
    % Generate a unit circle for ellipse fitting
    circleIn2D      = UnitCircleGenerate(nThetaEllipse);

    % Truncate the reference RGB to the specified dimensions
    rgb_ref_trunc   = rgb_ref(idx_varyingDim)';

    % Compute or use provided comparison stimuli
    if isempty(rgb_comp)
        % Compute the comparison stimuli if not provided
        rgb_comp_unscaled = rgb_ref_trunc + repmat(vecLen,[1,2]).*grid_theta_xy';
        rgb_comp_scaled = rgb_ref_trunc + repmat(vecLen,[1,2]).*ellipse_scaler.*grid_theta_xy';
        % Compute covariance of the unscaled comparison stimuli
        rgb_contour_cov = cov(rgb_comp_unscaled);
    else
        % Use provided comparison stimuli
        rgb_comp_trunc = rgb_comp(idx_varyingDim,:)';
        rgb_comp_scaled = rgb_ref_trunc + rgb_comp_trunc.*ellipse_scaler;
        rgb_contour_cov = cov(rgb_comp_trunc);
        rgb_comp_unscaled = rgb_comp_trunc;
    end

    % Fit an ellipse using iterative optimization
    % Initialize a high error value for comparison
    fitErr = 100;
    
    % Perform the fitting process 10 times with random initializations
    for n = 1:nRepeats
        % Generate random initial parameters:
        % - Two radii (random values between 50 and 1000)
        % - One rotation angle (random value between 0 and 360 degrees)
        init = [randi([50, 1000], 1), randi([50, 1000], 1), randi([0, 360], 1)];
        
        % Fit the ellipse using the FitEllipseQ function
        % Inputs:
        % - rgb_comp_unscaled': Comparison RGB values (transposed)
        % - rgb_ref_trunc': Reference RGB values (transposed)
        % - 'lockAngleAt0': Boolean flag to allow angle optimization
        % - 'ratioMax': Maximum ratio of ellipse axes (set to 3000)
        % - 'initialParams': Initial parameters for fitting (radii and angle)
        [ellParams_n, AConstraint_n, Ainv_n, Q_n, fitErr_n] = FitEllipseQ( ...
            rgb_comp_unscaled' - rgb_ref_trunc', ...
            'lockAngleAt0', false, ...
            'ratioMax', 3000, ...
            'initialParams', init);  
        
        % Update the best fit parameters if the current fit error is smaller
        if fitErr_n < fitErr
            fitErr = fitErr_n;        % Update the minimum fit error
            ellParams = ellParams_n;  % Best-fit ellipse parameters
            AConstraint = AConstraint_n; % Constraint matrix of the best fit
            Ainv = Ainv_n;            % Inverse constraint matrix
            Q = Q_n;                  % Quadratic form matrix of the ellipse
        end
    end

    % Adjust the fitted ellipse based on the reference stimulus
    fitEllipse_unscaled = (PointsOnEllipseQ(Q, circleIn2D) + rgb_ref_trunc')';
    % Scale the fitted ellipse
    fitEllipse_scaled = (fitEllipse_unscaled - rgb_ref_trunc).*...
        ellipse_scaler + rgb_ref_trunc;
end