function [deltaE, comp_RGB] = compute_deltaE(vecLen, background_RGB, ...
    ref_RGB, ref_Lab, vecDir, param, coloralg)
% Computes the perceptual difference (deltaE) between a reference stimulus
% and a comparison stimulus in the CIELab color space. The comparison stimulus
% is derived based on a specified chromatic direction and length from the reference.
%
% Inputs:
%   vecLen          - Scalar. The length to move in the specified direction from the reference stimulus.
%   background_RGB  - Vector [1x3]. The RGB values of the background, used in the conversion process.
%   ref_RGB         - Vector [1x3]. The RGB values of the reference stimulus.
%   ref_Lab         - Vector [1x3]. The CIELab values of the reference stimulus.
%   vecDir          - Vector [1x3]. The direction vector along which the comparison stimulus varies from the reference.
%   param           - Structure. Contains parameters necessary for RGB-to-Lab conversion:
%                     - param.B_monitor
%                     - param.T_cones
%                     - param.M_LMSToXYZ
%   coloralg        - String. Specifies the color difference formula to use:
%                     'CIE1976' (Euclidean), 'CIE94', or 'CIEDE2000'.
%
% Outputs:
%   deltaE          - Scalar. The computed perceptual difference between the reference and comparison stimuli.
%   comp_RGB        - Vector [1x3]. The RGB values of the comparison stimulus.

    % Validate and set default value for `coloralg`
    if nargin < 7 || isempty(coloralg)
        coloralg = 'CIE1976'; % Default to CIE1976 (Euclidean)
    elseif ~ismember(coloralg, {'CIE1976', 'CIE94', 'CIEDE2000'})
        error("Invalid 'coloralg' input. Valid options are 'CIE1976', 'CIE94', or 'CIEDE2000'.");
    end

    % Compute the RGB values for the comparison stimulus by moving along
    % the specified chromatic direction from the reference stimulus by vecLen.
    comp_RGB = ref_RGB + vecDir.*vecLen;

    % Convert the computed RGB values of the comparison stimulus into Lab values
    % using the provided parameters and the background RGB.
    [comp_Lab, ~, ~] = convert_rgb_lab(param.B_monitor,background_RGB,...
        param.T_cones, param.M_LMSToXYZ, comp_RGB);

    % Compute the perceptual difference (deltaE) between the stimuli
    if strcmp(coloralg, 'CIE1976')
        % Use Euclidean distance for CIE1976
        deltaE = norm(comp_Lab - ref_Lab);
    else
        % Use MATLAB's imcolordiff for CIE94 or CIEDE2000
        deltaE = imcolordiff(comp_Lab', ref_Lab', 'isInputLab', true, 'Standard', coloralg);
    end
end