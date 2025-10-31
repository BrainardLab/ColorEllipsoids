function [boundaryPoints, cornerPoints] = planeCubeBoundarySample(v1, v2, origin, varargin)
% planeCubeBoundarySample
%   Find intersection points between a plane (through 'origin' and spanned
%   by vectors v1, v2) and the boundary of an axis-aligned cube, using
%   radial sampling within the plane.
%
% Inputs
%   v1, v2  : (3x1 or 1x3) spanning vectors of the plane
%   origin  : (3x1 or 1x3) point on the plane; also the ray start
%
% Name–Value pairs (all optional, defaults shown)
%   'CubeBounds' : [0 1]     % cube is [lo, hi]^3
%   'NAngles'    : 1000      % number of in-plane ray directions
%   'Tol'        : 1e-4      % boundary/corner proximity tolerance
%   'ScaleMinMax': [0.1 2]   % radial scaling range along each ray
%   'NScaleSteps': 1e5       % number of steps along each ray
%
% Outputs
%   boundaryPoints : (<= nAngles x 3) boundary hits along in-plane rays
%   cornerPoints   : (4 x 3) points near XY-corners:
%                    [lo,lo,*], [lo,hi,*], [hi,lo,*], [hi,hi,*]
%
% Notes
%   - This implements your sampling approach: shoot many directions in the
%     plane, sample outward, and record the first contact with the cube.
%   - Corner detection looks for (x,y) ≈ the four XY-corners (z is free).
%   - For a cube centered at the origin, pass 'CubeBounds',[-1 1].

    % ---------------------------
    % Parse & validate inputs
    % ---------------------------
    p = inputParser;
    p.FunctionName = mfilename;

    addParameter(p, 'CubeBounds',  [0 1],    @(x)isnumeric(x)&&isequal(size(x),[1 2]));
    addParameter(p, 'NAngles',     1000,     @(x)isnumeric(x)&&isscalar(x)&&x>=3);
    addParameter(p, 'Tol',         1e-4,     @(x)isnumeric(x)&&isscalar(x)&&x>0);
    addParameter(p, 'ScaleMinMax', [0.1 2],  @(x)isnumeric(x)&&isequal(size(x),[1 2])&&x(2)>x(1));
    addParameter(p, 'NScaleSteps', 1e5,      @(x)isnumeric(x)&&isscalar(x)&&x>=10);

    parse(p, varargin{:});
    cubeBounds   = p.Results.CubeBounds;
    nAngles      = double(p.Results.NAngles);
    tol          = p.Results.Tol;
    scaleMinMax  = p.Results.ScaleMinMax;
    nScaleSteps  = double(p.Results.NScaleSteps);

    % Ensure column vectors
    v1     = v1(:);
    v2     = v2(:);
    origin = origin(:);

    % ---------------------------
    % 1) Orthonormal in-plane directions (uniform in angle)
    % ---------------------------
    rayDirs = samplePlaneDirections_(v1, v2, nAngles);   % (nAngles x 3)

    % ---------------------------
    % 2) March along each ray and record first boundary hit
    % ---------------------------
    boundaryPoints = NaN(nAngles, 3);
    scales = linspace(scaleMinMax(1), scaleMinMax(2), nScaleSteps);   % 1 x S

    lo = cubeBounds(1); hi = cubeBounds(2);

    for k = 1:nAngles
        dir3 = rayDirs(k, :);           % 1 x 3
        % origin + dir * scale (3 x S) via implicit expansion
        sampled = origin + dir3(:) .* scales;  % 3 x S

        % For each coord, find first index that hits lo or hi (within tol)
        hitX = find( abs(sampled(1,:) - lo) < tol | abs(sampled(1,:) - hi) < tol, 1, 'first' );
        hitY = find( abs(sampled(2,:) - lo) < tol | abs(sampled(2,:) - hi) < tol, 1, 'first' );
        hitZ = find( abs(sampled(3,:) - lo) < tol | abs(sampled(3,:) - hi) < tol, 1, 'first' );

        % Earliest contact index along this ray
        firstHitIdx = min([hitX, hitY, hitZ]);

        if ~isempty(firstHitIdx)
            boundaryPoints(k, :) = sampled(:, firstHitIdx).';
        end
    end

    % Drop rays that never hit (robustness)
    boundaryPoints = boundaryPoints(~any(isnan(boundaryPoints),2), :);

    % ---------------------------
    % 3) Find points near the four XY-corners (z free)
    % ---------------------------
    xyCornerTargets = [lo, lo;
                       lo, hi;
                       hi, hi;
                       hi, lo];
    cornerPoints = NaN(size(xyCornerTargets,1), 3);
    if ~isempty(boundaryPoints)
        % Compute L1 distance in XY to each target; pick any within tol*100
        XY = boundaryPoints(:,1:2);
        for c = 1:size(xyCornerTargets,1)
            idx = find( sum(abs(XY - xyCornerTargets(c,:)), 2) < tol*100, 1, 'first' );
            if ~isempty(idx)
                cornerPoints(c,:) = boundaryPoints(idx,:);
            end
        end
    end
end


function dirs = samplePlaneDirections_(v1, v2, nSamples)
% Create nSamples unit vectors uniformly spaced in angle within the plane
% spanned by v1 and v2 (Gram–Schmidt orthonormalization).
    e1 = v1 / norm(v1);
    v2p = v2 - (e1' * v2) * e1;
    nv2 = norm(v2p);
    if nv2 < 1e-14
        error('v1 and v2 are nearly collinear; cannot span a plane.');
    end
    e2 = v2p / nv2;

    theta = linspace(0, 2*pi, nSamples+1).';
    theta(end) = [];                % drop duplicate 2π

    % d(θ) = cosθ * e1 + sinθ * e2, unit directions in-plane
    dirs = cos(theta).*e1.' + sin(theta).*e2.';   % nSamples x 3
end

