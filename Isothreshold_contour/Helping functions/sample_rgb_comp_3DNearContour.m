function rgb_comp_sim = sample_rgb_comp_3DNearContour(rgb_ref,...
    nSims, radii, eigenVec, jitter)
% sample_rgb_comp_3DNearContour Generates simulated RGB components around an ellipsoid
%
% Inputs:
%   rgb_ref (3x1 vector): Reference RGB value around which the ellipsoid is centered.
%   nSims (integer): Number of simulations, i.e., number of points to generate.
%   radii (3x1 vector): Semi-axis lengths of the ellipsoid along its principal axes.
%   eigenVec (3x3 matrix): Eigenvectors defining the orientation of the ellipsoid's axes.
%   jitter (scalar): Standard deviation of the Gaussian noise added to points
%                    to simulate proximity to the ellipsoid surface.
%
% Output:
%   rgb_comp_sim (3xnSims matrix): Simulated RGB components. Each column represents
%                                  an RGB value near the ellipsoid contour.

    % Generate random angles for spherical coordinates. `randtheta` represents
    % angles in the XY plane, while `randphi` represents angles from the Z axis.
    % Uniformly distributed angles between 0 and 2*pi
    randtheta = rand(1,nSims).*2.*pi; 
    % Uniformly distributed angles between 0 and pi
    randphi   = rand(1,nSims).*pi;

    % Generate random points on the surface of a unit sphere by converting
    % spherical coordinates to Cartesian coordinates, then add Gaussian noise
    % (jitter) to each coordinate to simulate points near the surface.
    randx = sin(randphi).*cos(randtheta) + randn(1,nSims).*jitter;
    randy = sin(randphi).*sin(randtheta) + randn(1,nSims).*jitter;
    randz = cos(randphi) + randn(1,nSims).*jitter;

    % Stretch the random points by the ellipsoid's semi-axes lengths to fit
    % the ellipsoid's shape. This effectively scales the unit sphere points
    % to the size of the ellipsoid along each principal axis.
    randx_stretched = randx.*radii(1);
    randy_stretched = randy.*radii(2);
    randz_stretched = randz.*radii(3);

    % Combine the stretched coordinates into a single matrix. Each column
    % represents the (x, y, z) coordinates of a point.
    xyz = [randx_stretched; randy_stretched; randz_stretched];

    % Rotate and translate the simulated points to their correct positions
    % in RGB space. The rotation is defined by the ellipsoid's eigenvectors
    % (orientation), and the translation moves the ellipsoid to be centered
    % at the reference RGB value. This step aligns the ellipsoid with its
    % proper orientation and position as defined by the input parameters.
    rgb_comp_sim = eigenVec * xyz + rgb_ref;
    
end