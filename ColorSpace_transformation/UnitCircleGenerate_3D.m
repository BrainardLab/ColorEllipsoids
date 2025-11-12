function ellipsoids = UnitCircleGenerate_3D(nTheta, nPhi)
% Generates points on the surface of a unit sphere using spherical coords.
% Returns ellipsoids of size (nPhi, nTheta, 3) with x,y,z in the 3rd dim.

    % Angles (note: including both 0 and 2π duplicates first/last column)
    theta = linspace(0, 2*pi, nTheta);   % azimuth
    phi   = linspace(0, pi,    nPhi);    % polar (0=north pole)

    % Grids: THETA, PHI are (nPhi x nTheta)
    [THETA, PHI] = meshgrid(theta, phi);

    % Cartesian coords on unit sphere
    xCoords = sin(PHI) .* cos(THETA);
    yCoords = sin(PHI) .* sin(THETA);
    zCoords = cos(PHI);

    % Stack along 3rd dimension -> (nPhi x nTheta x 3)
    ellipsoids = cat(3, xCoords, yCoords, zCoords);
end