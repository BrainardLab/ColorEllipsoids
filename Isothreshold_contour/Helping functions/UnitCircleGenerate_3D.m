function ellipsoids = UnitCircleGenerate_3D(nTheta, nPhi)
    theta = linspace(0, 2*pi, nTheta);
    phi = linspace(0, pi, nPhi);

    [THETA, PHI] = meshgrid(theta, phi);
    xCoords = sin(PHI).*cos(THETA);
    yCoords = sin(PHI).*sin(THETA);
    zCoords = cos(PHI);

    ellipsoids = NaN(nPhi,nTheta,3);
    ellipsoids(:,:,1) = xCoords;
    ellipsoids(:,:,2) = yCoords;
    ellipsoids(:,:,3) = zCoords;
end