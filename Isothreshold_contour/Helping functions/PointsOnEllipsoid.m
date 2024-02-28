function ellipsoid = PointsOnEllipsoid(radii, center, eigenVectors, unitEllipsoid)

    x_Ellipsoid = squeeze(unitEllipsoid(:,:,1));
    y_Ellipsoid = squeeze(unitEllipsoid(:,:,2));
    z_Ellipsoid = squeeze(unitEllipsoid(:,:,3));

    x_stretched = x_Ellipsoid.*radii(1);
    y_stretched = y_Ellipsoid.*radii(2);
    z_stretched = z_Ellipsoid.*radii(3);

    xyz_rotated = eigenVectors*[x_stretched(:),y_stretched(:),z_stretched(:)]';

    ellipsoid = (xyz_rotated + center)';
end