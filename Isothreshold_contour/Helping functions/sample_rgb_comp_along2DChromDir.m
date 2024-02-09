function rgb_comp_sim = sample_rgb_comp_along2DChromDir(rgb_ref,...
    varying_RGBplane, slc_fixedVal, grid_theta_xy, nSims_perDir, vecLength, cov)

    numDirPts      = size(grid_theta_xy,2);
    fixed_RGBplane = setdiff(1:3, varying_RGBplane);
    nSims          = numDirPts*nSims_perDir;
    rgb_comp       = NaN(2, numDirPts, nSims_perDir);
    for d = 1:numDirPts
        cov_proj = cov * grid_theta_xy(:,d);
        noise = randn(2, nSims_perDir).*cov_proj;
        rgb_comp(:,d,:) = rgb_ref + grid_theta_xy(:,d).*vecLength(d) + noise;
    end
    rgb_comp_sim = NaN(3, nSims);
    rgb_comp_sim(fixed_RGBplane,:) = slc_fixedVal;
    rgb_comp1 = rgb_comp(1,:,:); 
    rgb_comp2 = rgb_comp(2,:,:);

    rgb_comp_sim(varying_RGBplane(1),:) = rgb_comp1(:);
    rgb_comp_sim(varying_RGBplane(2),:) = rgb_comp2(:);
end