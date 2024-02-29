function rgb_comp_sim = sample_rgb_comp_3DNearContour(rgb_ref,...
    nSims, radii, eigenVec, jitter)


    % Generate random angles to simulate points around the ellipse.
    randtheta = rand(1,nSims).*2.*pi; 
    randphi   = rand(1,nSims).*pi;
    % Calculate x and y coordinates with added jitter.
    randx = sin(randphi).*cos(randtheta) + randn(1,nSims).*jitter;
    randy = sin(randphi).*sin(randtheta) + randn(1,nSims).*jitter;
    randz = cos(randphi) + randn(1,nSims).*jitter;
    % Adjust coordinates based on the ellipsoid's semi-axis lengths.
    randx_stretched = randx.*radii(1);
    randy_stretched = randy.*radii(2);
    randz_stretched = randz.*radii(3);
    % Calculate the varying RGB dimensions, applying rotation and 
    % translation based on the reference RGB values and ellipsoid 
    % parameters.
    xyz = [randx_stretched; randy_stretched; randz_stretched];

    rgb_comp_sim = eigenVec * xyz + rgb_ref;
    
end