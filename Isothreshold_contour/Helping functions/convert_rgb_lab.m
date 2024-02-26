function [color_Lab, color_XYZ, color_LMS] = convert_rgb_lab(monitor_Spd,...
    background_RGB, T_cones, M_LMSToXYZ, color_RGB)
% This function converts an RGB color value into the CIELab color space
% using the monitor's spectral power distribution (SPD), the background RGB
% values, cone sensitivities (T_cones), and a matrix that converts from LMS
% (cone responses) to CIEXYZ color space (M_LMSToXYZ).
%
% Inputs:
%   monitor_Spd - Spectral power distribution of the monitor.
%   background_RGB - Background RGB values used for normalization.
%   T_cones - Matrix of cone sensitivities for absorbing photons at different wavelengths.
%   M_LMSToXYZ - Matrix to convert LMS cone responses to CIEXYZ.
%   color_RGB - RGB color value(s) to be converted.
%
% Outputs:
%   color_Lab - The converted color(s) in CIELab color space.
%   color_XYZ - The converted color(s) in CIEXYZ color space.
%   color_LMS - The LMS cone response for the given color_RGB.

    % Convert background RGB to SPD using the monitor's SPD
    background_Spd = monitor_Spd*background_RGB;
    % Convert background SPD to LMS (cone response)
    background_LMS = T_cones*background_Spd;
    % Convert background LMS to XYZ (for use in Lab conversion)
    background_XYZ = M_LMSToXYZ*background_LMS;

    %RGB -> SPD
    color_Spd      = monitor_Spd*color_RGB;
    %SPD -> LMS
    color_LMS      = T_cones*color_Spd;
    %LMS -> XYZ
    color_XYZ      = M_LMSToXYZ*color_LMS;
    %XYZ -> Lab
    color_Lab      = XYZToLab(color_XYZ, background_XYZ);
end