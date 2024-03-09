function ellipsoid = PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid)
% This function computes points on the surface of an ellipsoid given its
% radii, center, orientation (via eigenVectors), and a unit ellipsoid
% (essentially a scaled unit sphere).

    % Extract the x, y, and z coordinates of points on the unit ellipsoid's
    % surface. The unitEllipsoid input is a 3D array where each layer along
    % the third dimension represents the x, y, and z coordinates, respectively.
    x_Ellipsoid = squeeze(unitEllipsoid(:,:,1));
    y_Ellipsoid = squeeze(unitEllipsoid(:,:,2));
    z_Ellipsoid = squeeze(unitEllipsoid(:,:,3));

    % Scale the x, y, and z coordinates by the ellipsoid's radii to stretch
    % the unit sphere into the ellipsoid shape. The .* operator is used for
    % element-wise multiplication of the coordinates with the corresponding
    % radii.
    x_stretched = x_Ellipsoid.*radii(1);
    y_stretched = y_Ellipsoid.*radii(2);
    z_stretched = z_Ellipsoid.*radii(3);

    %To rotate your ellipsoid to align with the eigenvectors, you should 
    % perform the multiplication with the eigenvectors matrix evecs on the 
    % left and your coordinates matrix xyz on the right, where xyz is 
    % organized as a 3 by N matrix. This is because each column of the 
    % evecs matrix represents an eigenvector corresponding to one of the 
    % principal axes of the ellipsoid
    xyz_rotated = eigenVectors*[x_stretched(:),y_stretched(:),z_stretched(:)]';

    % Translate the rotated coordinates by adding the center of the ellipsoid.
    % The center is added to each column of the rotated coordinates matrix,
    % effectively moving the ellipsoid to its correct position in space.
    % The result is transposed back to have rows as individual x, y, z
    % coordinates for consistency with the input format.
    ellipsoid = (xyz_rotated + center)';
end