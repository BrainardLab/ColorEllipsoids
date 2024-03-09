function ellipsoids = UnitCircleGenerate_3D(nTheta, nPhi)
% UnitCircleGenerate_3D generates a unit ellipsoid's coordinates in 3D space.
% 
% Inputs:
% - nTheta: Number of divisions or points along the theta dimension.
%           It controls the resolution around the z-axis.
% - nPhi: Number of divisions or points along the phi dimension.
%         It controls the resolution from top to bottom (north to south pole).
%
% Outputs:
% - ellipsoids: A 3D matrix containing the x, y, and z coordinates of the points
%               on the surface of the unit ellipsoid. The size of the matrix
%               is nPhi x nTheta x 3, where the third dimension corresponds to
%               the x, y, and z coordinates, respectively.

    % Generate linearly spaced vectors for theta (0 to 2*pi) and phi (0 to pi).
    % Theta ranges from 0 to 2*pi, covering 360 degrees around the z-axis.
    % Phi ranges from 0 to pi, covering 180 degrees from the north to the south pole.
    theta = linspace(0, 2*pi, nTheta);
    phi = linspace(0, pi, nPhi);

    % Generate a grid of theta and phi values using meshgrid. This step is crucial for
    % vectorized calculations of x, y, and z coordinates. THETA and PHI are matrices
    % where each element corresponds to a pair of theta and phi values for a point
    % on the ellipsoid.
    [THETA, PHI] = meshgrid(theta, phi);

    % Calculate the x, y, and z coordinates of points on the unit ellipsoid
    % surface using the spherical to Cartesian coordinates conversion formula.
    xCoords = sin(PHI).*cos(THETA);
    yCoords = sin(PHI).*sin(THETA);
    zCoords = cos(PHI);

    % Assign the calculated x, y, and z coordinates to their respective layers
    % in the ellipsoids matrix. This organization makes it easy to access all x, y,
    % or z coordinates by selecting the appropriate page of the matrix.
    ellipsoids = NaN(nPhi,nTheta,3);
    ellipsoids(:,:,1) = xCoords;
    ellipsoids(:,:,2) = yCoords;
    ellipsoids(:,:,3) = zCoords;
end