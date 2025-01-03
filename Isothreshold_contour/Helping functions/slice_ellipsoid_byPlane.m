function [sliced_ellipse, details] = slice_ellipsoid_byPlane(center, radii, eigenvectors, plane_v1, plane_v2, covMat, num_grid_pts)
    % Computes the intersection of an ellipsoid with a plane, resulting in an elliptical contour.
    %
    % Parameters:
    % - center: A 3x1 vector representing the center of the ellipsoid.
    % - radii: A 3x1 vector representing the semi-axes (radii) of the ellipsoid.
    % - eigenvectors: A 3x3 matrix where each column is an eigenvector defining the orientation of the ellipsoid.
    % - plane_v1: A 3x1 vector representing the first vector that lies on the plane.
    % - plane_v2: A 3x1 vector representing the second vector that lies on the plane.
    % - covMat: (Optional) 3x3 covariance matrix for the ellipsoid. If not provided, it will be computed.
    % - num_grid_pts: (Optional) Number of points to represent the ellipse.
    %
    % Returns:
    % - sliced_ellipse: A 3xN matrix representing the 3D coordinates of the elliptical contour.
    % - details: A struct containing additional information (M, eigenvalues, eigenvectors, semi-axes).

    if nargin < 6 || isempty(covMat)
        % Construct the matrix A for the ellipsoid equation
        % The ellipsoid is defined by the equation: (x - center)^T A (x - center) = 1
        % where A = R * D^(-2) * R^T, with R being the rotation matrix (eigenvectors)
        % and D being the diagonal matrix of radii
        A = eigenvectors * diag(1 ./ (radii .^ 2)) * eigenvectors';
    else
        A = covMat;
    end

    if nargin < 7
        num_grid_pts = 100;
    end

    % Normalize the input vectors that define the plane
    v1 = plane_v1 / norm(plane_v1);
    v2 = plane_v2 / norm(plane_v2);

    % Check if the two vectors defining the plane are orthogonal
    if abs(dot(v1, v2)) > 1e-3
        error('The two vectors defining the plane should be orthogonal!');
    end

    % Compute the quadratic form in the plane's local coordinate system
    % M is a 2x2 matrix that represents the quadratic form of the ellipsoid equation
    % restricted to the plane spanned by v1 and v2
    M = [v1' * A * v1, v1' * A * v2; v2' * A * v1, v2' * A * v2];

    % Eigendecomposition of M to obtain the semi-axes and rotation of the ellipse
    [eigvecs, eigvals_diag] = eig(M);
    eigvals = diag(eigvals_diag);

    % The lengths of the semi-axes of the ellipse are the inverses of the square roots of the eigenvalues
    semi_axes = 1 ./ sqrt(eigvals);

    % Parametrize the ellipse in the plane's local coordinate system
    % The ellipse is parameterized using an angle from 0 to 2*pi
    angles = linspace(0, 2 * pi, num_grid_pts);
    ellipse_local = [semi_axes(1) * cos(angles); semi_axes(2) * sin(angles)];

    % Rotate the ellipse to align with the correct orientation in the plane
    ellipse_local_rotated = eigvecs * ellipse_local;

    % Transform the ellipse from the plane's local coordinates to global 3D coordinates
    % This step places the ellipse in the global coordinate system by using the plane's basis vectors (v1, v2)
    sliced_ellipse = center' + repmat(ellipse_local_rotated(1, :)',[1,3]) .* repmat(v1',[num_grid_pts, 1])  +...
        repmat(ellipse_local_rotated(2, :)',[1,3]) .* repmat(v2',[num_grid_pts, 1]) ;

    % Reshape sliced_ellipse to be 3xN
    sliced_ellipse = sliced_ellipse';

    % Store additional details
    details.M = M;
    details.eigvals = eigvals;
    details.eigvecs = eigvecs;
    details.semi_axes = semi_axes;
end
