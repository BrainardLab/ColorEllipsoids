function rgb_comp_sim = sample_rgb_comp_along2DChromDir(rgb_ref,...
    varying_RGBplane, slc_fixedVal, nSims, paramEllipse, jitter)

    fixed_RGBplane = setdiff(1:3, varying_RGBplane);
    rgb_comp_sim = NaN(3, nSims);

    randtheta = rand(1,nSims).*2.*pi; 
    randx = cos(randtheta) + randn(1,nSims).*jitter;
    randy = sin(randtheta) + randn(1,nSims).*jitter;
    randx_stretched = randx./paramEllipse(1);
    randy_stretched = randy./paramEllipse(2);
    rgb_comp_sim(varying_RGBplane(1),:) = ...
        randx_stretched.*cosd(paramEllipse(3)) - ...
        randy_stretched.*sind(paramEllipse(3)) + rgb_ref(1);
    rgb_comp_sim(varying_RGBplane(2),:) = ...
        randx_stretched.*sind(paramEllipse(3)) + ...
        randy_stretched.*cosd(paramEllipse(3)) + rgb_ref(2);

    rgb_comp_sim(fixed_RGBplane,:) = slc_fixedVal;
end