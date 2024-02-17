function pIncorrect = predict_error_prob_individualEllipses(ellParam,...
    x_ref, x_comp, alpha_Weibull, beta_Weibull)
    %x_ref: size of 2 by 1
    %x_comp: size of 2 by N
    majorAxis = 1/ellParam(1);
    minorAxis = 1/ellParam(2);
    rotDeg    = ellParam(3);

    %rotation matrix
    rot_M = [cosd(rotDeg), -sind(rotDeg); sind(rotDeg), cosd(rotDeg)];

    %make them centered around 0
    x_comp_centered = x_comp - x_ref;
    %unrotate them
    x_comp_centered_unrotated = inv(rot_M)*x_comp_centered; 
    %unstretch them
    x_comp_centered_unrotated_unstretched = x_comp_centered_unrotated./[majorAxis;minorAxis];

    %visualize it
    % figure; scatter(x_comp_centered_unrotated_unstretched(1,:), x_comp_centered_unrotated_unstretched(2,:)); axis equal;axis square
    
    %now x_comp should be like a unit circle 
    L2_x_comp = sqrt(x_comp_centered_unrotated_unstretched(1,:).^2 +...
        x_comp_centered_unrotated_unstretched(2,:).^2);
    pIncorrect = 1 - ComputeWeibTAFC(L2_x_comp, alpha_Weibull, beta_Weibull);
end