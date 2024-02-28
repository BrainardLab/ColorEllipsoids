function rgb_comp_sim = sample_rgb_comp_2DNearContour(rgb_ref,...
    varying_RGBplane, slc_fixedVal, nSims, paramEllipse, jitter)
% Samples RGB compositions near an isothreshold ellipsoidal contour.
%
% This function generates simulated RGB compositions based on a reference
% RGB value and parameters defining an ellipsoidal contour. The function
% is designed to vary two of the RGB dimensions while keeping the third fixed,
% simulating points near the contour of an ellipsoid in RGB color space.
%
% Inputs:
%   rgb_ref: 3-element vector specifying the reference RGB composition.
%   varying_RGBplane: 2-element vector indicating which RGB dimensions to vary.
%   slc_fixedVal: Scalar value for the fixed RGB dimension.
%   nSims: Number of simulations to generate.
%   paramEllipse: 3-element vector specifying the ellipsoid parameters.
%                 [a b theta], where a and b are the semi-axis lengths and
%                 theta is the rotation angle in degrees.
%   jitter: Scalar value specifying the amount of random jitter to add,
%           simulating variability around the ellipsoidal contour. Noise is
%           assumed to be Gaussian.
%
% Outputs:
%   rgb_comp_sim: 3-by-nSims matrix of simulated RGB compositions. Each column
%                 represents a simulated RGB composition.

    % Identify the fixed RGB dimension by excluding the varying dimensions.
    fixed_RGBplane = setdiff(1:3, varying_RGBplane);
    % Initialize the output matrix with NaNs.
    rgb_comp_sim = NaN(3, nSims);

    % Generate random angles to simulate points around the ellipse.
    randtheta = rand(1,nSims).*2.*pi; 
    % Calculate x and y coordinates with added jitter.
    randx = cos(randtheta) + randn(1,nSims).*jitter;
    randy = sin(randtheta) + randn(1,nSims).*jitter;
    % Adjust coordinates based on the ellipsoid's semi-axis lengths.
    randx_stretched = randx./paramEllipse(1);
    randy_stretched = randy./paramEllipse(2);
    % Calculate the varying RGB dimensions, applying rotation and 
    % translation based on the reference RGB values and ellipsoid 
    % parameters.
    rgb_comp_sim(varying_RGBplane(1),:) = ...
        randx_stretched.*cosd(paramEllipse(3)) - ...
        randy_stretched.*sind(paramEllipse(3)) + rgb_ref(1);
    rgb_comp_sim(varying_RGBplane(2),:) = ...
        randx_stretched.*sind(paramEllipse(3)) + ...
        randy_stretched.*cosd(paramEllipse(3)) + rgb_ref(2);
    
    % Set the fixed RGB dimension to the specified fixed value for all 
    % simulations.
    rgb_comp_sim(fixed_RGBplane,:) = slc_fixedVal;
end