function [color_Lab, color_XYZ, color_LMS] = convert_rgb_lab(monitor_Spd,...
    background_RGB, T_cones, M_LMSToXYZ, color_RGB)

    background_Spd = monitor_Spd*background_RGB;
    background_LMS = T_cones*background_Spd;
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