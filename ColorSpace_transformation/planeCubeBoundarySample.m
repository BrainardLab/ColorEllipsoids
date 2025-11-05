function [boundaryPoints, cornerPoints, e1, e2] = planeCubeBoundarySample(v1, v2, origin, varargin)
% planeCubeBoundarySample
%   Approximate the intersection curve between an axis-aligned cube and a
%   plane (defined by a point 'origin' and spanning vectors v1, v2) by
%   radial sampling within the plane.
%
% Inputs
%   v1, v2    : (3x1 or 1x3) spanning vectors of the plane
%   origin    : (3x1 or 1x3) point on the plane; also the ray start
%
% Name–Value pairs (optional; defaults shown)
%   'CubeBounds'      : [0 1]        % cube is [lo, hi]^3 (axis-aligned)
%   'NAngles'         : 1e5          % number of in-plane ray directions
%   'intersectionDims': [1, 2]       % which dims define the "corner" plane
%                                    % e.g., [1,2] → XY, [1,3] → XZ, [2,3] → YZ
%
% Outputs
%   boundaryPoints : (<= NAngles x 3) first boundary hit per in-plane ray
%   cornerPoints   : (4 x 3) points nearest to the four 2D “corners” in the
%                    selected dims (z is free if dims=[1,2], etc.).
%
% Method (sampling-based, approximate)
%   1) Build an orthonormal basis {e1, e2} for the plane via Gram–Schmidt.
%   2) For each angle θ, shoot a unit ray d(θ) = cosθ e1 + sinθ e2 from 'origin'.
%   3) For each ray, compute parametric scalars t to the six cube slabs
%      (x=lo/hi, y=lo/hi, z=lo/hi) as (bound - origin) ./ dir, ignore t<=0,
%      and select the smallest positive t as the first boundary contact.
%   4) Keep rays that produce a finite hit point (robustness cleanup).
%   5) Among the collected boundary samples, pick those nearest to the
%      four 2D corners (lo/hi pairs in 'intersectionDims') as cornerPoints.

    % ---------------------------
    % Parse & validate inputs
    % ---------------------------
    p = inputParser;
    p.FunctionName = mfilename;

    addParameter(p, 'CubeBounds',  [0 1],    @(x)isnumeric(x)&&isequal(size(x),[1 2]));
    addParameter(p, 'NAngles',     1e6,      @(x)isnumeric(x)&&isscalar(x)&&x>=3);
    addParameter(p, 'intersectionDims', [1,2], @(x)isnumeric(x)&&isequal(size(x),[1 2]));

    parse(p, varargin{:});
    cubeBounds       = p.Results.CubeBounds;
    nAngles          = double(p.Results.NAngles);
    intersectionDims = p.Results.intersectionDims;

    % Force column vectors
    v1     = v1(:);
    v2     = v2(:);
    origin = origin(:);

    % ---------------------------
    % 1) Orthonormal in-plane directions (uniform angular sampling)
    % ---------------------------
    [rayDirs, e1, e2] = samplePlaneDirections_(v1, v2, nAngles, 0.3);   % (NAngles x 3)

    % ---------------------------
    % 2) March along each ray; record first boundary hit
    %
    % For each ray component i∈{x,y,z}, t = (bound_i - origin_i) / dir_i.
    % We assemble all 6 slab candidates and take the smallest positive t.
    % ---------------------------
    boundaryPoints = NaN(nAngles, 3);
    lo = cubeBounds(1); hi = cubeBounds(2);

    for k = 1:nAngles
        dir3 = rayDirs(k, :);                     % 1 x 3
        % 3x2 array of t to the six slabs:
        %   row = axis (x,y,z), col = bound (lo,hi)
        scalers = ([lo, hi; lo, hi; lo, hi] - origin) ./ dir3(:);

        % Ignore non-advancing or invalid steps
        B = scalers;
        B(B <= 0) = Inf;

        % First positive intersection along the ray
        [val, ~] = min(B(:));

        % Store candidate hit point
        boundaryPoints(k,:) = origin + dir3(:) .* val;
    end

    % Discard rays with no valid hit (NaN rows)
    boundaryPoints = boundaryPoints(~any(isnan(boundaryPoints),2), :);

    % ---------------------------
    % 3) Select samples nearest to the four 2D “corners”
    %
    % We project boundaryPoints to the selected dims, and for each 2D
    % target corner (lo/hi pair), pick the sample with minimum L1 distance.
    % ---------------------------
    xyCornerTargets = [lo, lo;
                       lo, hi;
                       hi, hi;
                       hi, lo];
    cornerPoints = NaN(size(xyCornerTargets,1), 3);

    if ~isempty(boundaryPoints)
        XY = boundaryPoints(:, intersectionDims);  % project to chosen 2D plane
        for c = 1:size(xyCornerTargets,1)
            [~,idx] = min(sum(abs(XY - xyCornerTargets(c,:)), 2)); % L1
            if ~isempty(idx)
                cornerPoints(c,:) = boundaryPoints(idx,:);
            end
        end
    end
end


function [dirs, e1,e2] = samplePlaneDirections_(v1, v2, nSamples, scaler)
    if nargin < 4; scaler = 1; end
% samplePlaneDirections_
%   Build an orthonormal basis {e1,e2} for span(v1,v2) via Gram–Schmidt,
%   then return nSamples unit directions uniformly spaced in angle:
%     d(θ) = cosθ * e1 + sinθ * e2, for θ ∈ [0, 2π).
%
% Notes
%   - If v1 and v2 are nearly collinear, Gram–Schmidt may be ill-conditioned.
%     Provide two reasonably independent spanning vectors for robust behavior.

    [e1, e2] = gram_Schmidt(v1, v2);

    theta = linspace(0, 2*pi, nSamples+1).';
    theta(end) = [];                           % drop duplicate 2π endpoint

    e1 = scaler*e1;
    e2 = scaler*e2;

    % Produce a (nSamples x 3) matrix of unit directions in the plane
    dirs = cos(theta).*e1.' + sin(theta).*e2.';   % nSamples x 3
end

