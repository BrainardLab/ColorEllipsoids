function rgb_comp_sim = sample_rgb_comp_2Drandom(rgb_ref, varying_RGBplane,...
    slc_fixedVal, range, nSims)
    fixed_RGBplane = setdiff(1:3, varying_RGBplane);
    rgb_comp_sim   = rand(3, nSims).*(range(2)-range(1))+range(1);
    rgb_comp_sim(fixed_RGBplane,:)   = slc_fixedVal;
    rgb_comp_sim(varying_RGBplane,:) = rgb_comp_sim(varying_RGBplane,:) + rgb_ref;
end