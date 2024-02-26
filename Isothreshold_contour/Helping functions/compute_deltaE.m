function [deltaE, comp_RGB] = compute_deltaE(vecLen, background_RGB, ...
    ref_RGB, ref_Lab, vecDir, param)
% Computes the perceptual difference (deltaE) between a reference stimulus
% and a comparison stimulus in the CIELab color space. The comparison stimulus
% is derived based on a specified chromatic direction and length from the reference.
%
% Inputs:
%   vecLen          - The length to move in the specified direction from the reference stimulus.
%   background_RGB  - The RGB values of the background, used in the conversion process.
%   ref_RGB         - The RGB values of the reference stimulus.
%   ref_Lab         - The CIELab values of the reference stimulus.
%   vecDir          - The direction vector along which the comparison stimulus varies from the reference.
%   param           - A structure containing parameters necessary for RGB to Lab conversion, including:
%   
% Outputs:
%   deltaE          - The computed perceptual difference between the reference and comparison stimuli.
%   comp_RGB        - The RGB values of the comparison stimulus.

    % Compute the RGB values for the comparison stimulus by moving along
    % the specified chromatic direction from the reference stimulus by vecLen.
    comp_RGB = ref_RGB + vecDir.*vecLen;

    % Convert the computed RGB values of the comparison stimulus into Lab values
    % using the provided parameters and the background RGB.
    [comp_Lab, ~, ~] = convert_rgb_lab(param.B_monitor,background_RGB,...
        param.T_cones, param.M_LMSToXYZ, comp_RGB);

    % Calculate the perceptual difference (deltaE) between the reference and comparison
    % stimuli as the Euclidean distance between their Lab values.
    deltaE = norm(comp_Lab - ref_Lab);
end