function rgb_comp_sim = sample_rgb_comp_random(rgb_ref, varying_RGBplane,...
    slc_fixedVal, range, nSims)
% This function samples random points in a 2D/3D space within a specified 
% range for a square/cube. If one dim is fixed, the function wii generate
% samples round the isothreshold contour centered at the reference color,
% while keeping one of the RGB components fixed.

    % fixed_RGBplane: Identifies the RGB component that remains fixed.
    % It does this by finding the difference between the full set of RGB planes (1:3)
    % and the varying_RGBplane parameter provided to the function.
    fixed_RGBplane = setdiff(1:3, varying_RGBplane);

    % It generates random values for each of the three RGB components across
    % 'nSims' simulations. The random values are scaled to the specified 'range'.
    rgb_comp_sim   = rand(3, nSims).*(range(2)-range(1))+range(1);

    %if at least one plane is fixed
    if ~isempty(fixed_RGBplane)
        % This line sets the fixed RGB component(s) to the value specified 
        % by slc_fixedVal for all simulations. The fixed component remains 
        % constant across all simulations.
        rgb_comp_sim(fixed_RGBplane,:)   = slc_fixedVal;
    end

    % This line adjusts the varying RGB components by adding the reference 
    % RGB value (rgb_ref) to the randomly generated values. This ensures 
    % the variations are relative to the reference RGB value provided.
    rgb_comp_sim(varying_RGBplane,:) = rgb_comp_sim(varying_RGBplane,:) + rgb_ref;
end