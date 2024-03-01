function ellipsoid = PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid)

    x_Ellipsoid = squeeze(unitEllipsoid(:,:,1));
    y_Ellipsoid = squeeze(unitEllipsoid(:,:,2));
    z_Ellipsoid = squeeze(unitEllipsoid(:,:,3));

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

    ellipsoid = (xyz_rotated + center)';
end