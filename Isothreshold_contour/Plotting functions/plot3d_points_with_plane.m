function ax = plot3d_points_with_plane(X, varargin)
%PLOT3D_POINTS_WITH_PLANE Plot 3D points (3xN) with optional plane patch (from 3x4 corners).
%
%   ax = plot3d_points_with_plane(X, 'Name', Value, ...)
%
% Required:
%   X : 3xN matrix of coordinates
%
% Name-Value options:
%   'PlaneCorners' : 3x4 matrix of corner points to draw a convex-hull patch (default [])
%   'PointColors'  : Nx3 matrix of RGB colors for each point (default [])
%   'Labels'       : 1x3 cellstr or string array, e.g. {'R','G','B'} (default {'X','Y','Z'})
%   'Limits'       : 3x2 matrix [xmin xmax; ymin ymax; zmin zmax] (default [])
%   'View'         : 1x2 [az el] (default [35 25])
%   'Marker'       : marker symbol (default 'd')
%   'MarkerSize'   : scalar (default 50)
%   'Filled'       : true/false (default true)
%   'FaceColor'    : 1x3 RGB for plane (default [0.5 0.5 0.5])
%   'FaceAlpha'    : scalar (default 0.25)
%   'EdgeColor'    : edge color for plane (default 'none')
%   'AxisEqual'    : true/false (default true)

% ---- parse inputs ----
p = inputParser;
p.addRequired('X', @(v) isnumeric(v) && size(v,1)==3);

p.addParameter('PlaneCorners', [], @(v) isempty(v) || (isnumeric(v) && size(v,1)==3 && size(v,2)==4));
p.addParameter('PointColors', [], @(v) isempty(v) || (isnumeric(v) && size(v,2)==3));
p.addParameter('Labels', {'X','Y','Z'}, @(v) (iscell(v) || isstring(v)) && numel(v)==3);
p.addParameter('LabelInterpreter', 'none', @(s) ischar(s) || isstring(s));
p.addParameter('Limits', [], @(v) isempty(v) || (isnumeric(v) && isequal(size(v),[3 2])));
p.addParameter('View', [35 25], @(v) isnumeric(v) && numel(v)==2);

p.addParameter('Marker', 'd', @(v) ischar(v) || isstring(v));
p.addParameter('MarkerSize', 50, @(v) isnumeric(v) && isscalar(v));
p.addParameter('Filled', true, @(v) islogical(v) && isscalar(v));

p.addParameter('FaceColor', [0.5 0.5 0.5], @(v) isnumeric(v) && numel(v)==3);
p.addParameter('FaceAlpha', 0.25, @(v) isnumeric(v) && isscalar(v));
p.addParameter('EdgeColor', 'none', @(v) true);

p.addParameter('AxisEqual', true, @(v) islogical(v) && isscalar(v));

p.parse(X, varargin{:});
opt = p.Results;

% ---- setup ----
figure; ax = gca;
hold(ax,'on'); box(ax,'on'); grid(ax,'on');

% ---- plane patch (optional) ----
if ~isempty(opt.PlaneCorners)
    V = opt.PlaneCorners.'; % 4x3
    K = convhull(V(:,1), V(:,2), V(:,3));
    trisurf(K, V(:,1), V(:,2), V(:,3), ...
        'FaceColor', opt.FaceColor, 'FaceAlpha', opt.FaceAlpha, ...
        'EdgeColor', opt.EdgeColor, 'Parent', ax);
end

% ---- scatter points ----
x = X(1,:); y = X(2,:); z = X(3,:);

if isempty(opt.PointColors)
    if opt.Filled
        scatter3(ax, x, y, z, opt.MarkerSize, 'filled', 'Marker', opt.Marker);
    else
        scatter3(ax, x, y, z, opt.MarkerSize, 'Marker', opt.Marker);
    end
else
    C = opt.PointColors;
    if size(C,1) ~= numel(x)
        error('PointColors must be Nx3 where N = size(X,2).');
    end
    if opt.Filled
        scatter3(ax, x, y, z, opt.MarkerSize, C, 'filled', 'Marker', opt.Marker);
    else
        scatter3(ax, x, y, z, opt.MarkerSize, C, 'Marker', opt.Marker);
    end
end

% ---- labels ----
lbl = opt.Labels;
xlabel(ax, lbl{1}, 'Interpreter', opt.LabelInterpreter);
ylabel(ax, lbl{2}, 'Interpreter', opt.LabelInterpreter);
zlabel(ax, lbl{3}, 'Interpreter', opt.LabelInterpreter);

% ---- limits ----
if ~isempty(opt.Limits)
    xlim(ax, opt.Limits(1,:));
    ylim(ax, opt.Limits(2,:));
    zlim(ax, opt.Limits(3,:));
elseif ~isempty(opt.PlaneCorners)
    % sensible default: limits from plane corners
    V = opt.PlaneCorners.';
    xlim(ax, [min(V(:,1)) max(V(:,1))]);
    ylim(ax, [min(V(:,2)) max(V(:,2))]);
    zlim(ax, [min(V(:,3)) max(V(:,3))]);
end

% ---- appearance ----
if opt.AxisEqual, axis(ax,'equal'); end
view(ax, opt.View(1), opt.View(2));

end