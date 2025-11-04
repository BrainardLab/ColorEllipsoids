clear all; close all; clc
% -------------------------------------------------------------------------
% Goal
%   Build an LS cone–isolating plane for testing dichromats without M cones.
%   Pipeline mirrors extract_LMisolating_plane.m, and compares against the
%   (previous) isoluminant plane used in the eLife paper.
%
% High-level steps
%   1) Sample points on the LM cone-contrast plane (S = 0).
%   2) Map those points cc → LMS → RGB.
%   3) In RGB, intersect the plane with the display cube [0,1]^3 (polygon).
%   4) From the polygon’s 4 “corners,” compute the 2D↔RGB linear transform.
% -------------------------------------------------------------------------

% --- Load calibration and color-space transforms used in the adaptation expt
cal_path = fullfile(getpref('ColorEllipsoids','ELPSMaterials'),'Calibration');
S = load(fullfile(cal_path, 'Transformation_btw_color_spaces_10062025.mat'), ...
                  'DELL_10062025_texture_right');

% Background/ambient and transforms (all 3x1 or 3x3 as noted)
bg_LMS      = S.DELL_10062025_texture_right.bgLMS;         % 1x3 background LMS (row)
ambient_LMS = S.DELL_10062025_texture_right.ambientLMS;    % 3x1 ambient LMS (col)
M_RGBToLMS  = S.DELL_10062025_texture_right.M_RGBToLMS;    % 3x3 RGB→LMS
M_LMSToRGB  = S.DELL_10062025_texture_right.M_LMSTORGB;   % 3x3 LMS→RGB

% Keep the isoluminant mapping separate to avoid confusion with LS mapping
M_2DWToRGB_isoluminant = S.DELL_10062025_texture_right.M_2DWToRGB;
cornerPointsRGB_isoluminant = S.DELL_10062025_texture_right.cornerPointsRGB(:,[1,3,4,2])';

% Define the isolating plane metadata 
isolating_plane = 'LS';           % label for plots/titles
isolating_dims  = [1,3];          % use X=R (1) and Z=B (3) when finding corners
silencing_dim   = 2;              % M-cone (index 2) is silenced (contrast = 0)

% P_cc is 3×3 cone-contrast rows: [L M S].
% We pick exactly three points that lie on the LS plane (M=0):
%   • an L-only point, an S-only point, and the origin.
% Any two independent vectors from these three will span the entire LS (2D) plane.
P_cc = [0.1, 0,   0;    % +L,  M=0, S=0
        0,   0,   0;    % origin (bg)
        0,   0, 0.1];   % +S,  M=0

% --- Map cc → LMS → RGB (subtract ambient for device modeling)
LMS_3d = ContrastToExcitation(P_cc', bg_LMS) - ambient_LMS;  % (3×N)
Pts_RGB_all = (M_LMSToRGB * LMS_3d).';                       % (N×3), each row ∈ RGB

% --- Sanity check: back-transform RGB→LMS→cc and verify M-cone is ~0
LMS_3d_sanity          = M_RGBToLMS * Pts_RGB_all' + ambient_LMS;
coneContrast_3d_sanity = ExcitationToContrast(LMS_3d_sanity, bg_LMS);
if min(abs(coneContrast_3d_sanity(silencing_dim,:))) > 1e-4
    error('Sanity check failed: M-cone contrast not sufficiently silenced.');
end

%% Select an in-gamut subset and derive the plane→RGB transform from corners
% Keep only points well inside [0,1]^3 (strict bounds to avoid edge noise)
mask_bds = [0.15, 0.85];
mask = all(Pts_RGB_all > mask_bds(1) & Pts_RGB_all < mask_bds(2), 2);
idx = find(mask);

Pts_RGB          = Pts_RGB_all(idx, :);
Pts_coneContrast = coneContrast_3d_sanity(:,idx)';  % (Nsel×3), for reference/saving
Pts_LMS          = LMS_3d_sanity(:,idx)';

% Define two in-plane spanning vectors in RGB and the plane origin:
%   • origin: the mean of the selected points (robust to which 2 survive)
%   • v1: from origin to the first selected point
%   • v2: from origin to the last selected point
% These two vectors (unless collinear) span the LS plane in RGB space.
% planeCubeBoundarySample then shoots rays in this plane and finds the
% intersection polygon with the display cube.
Pts_RGB_mean = mean(Pts_RGB, 1);
[Pts_RGB_inGamut, Pts_RGB_corners] = planeCubeBoundarySample( ...
    Pts_RGB(1,:)  - Pts_RGB_mean, ...   % v1 in RGB plane
    Pts_RGB(end,:) - Pts_RGB_mean, ...  % v2 in RGB plane (distinct from v1)
    Pts_RGB_mean, ...                   % plane origin in RGB
    'intersectionDims', isolating_dims);

% --- Solve for 2D→RGB transform using 4 labeled plane points (±1,±1)
%     We treat plane coordinates as [w1, w2, 1] with w ∈ {−1, +1}.
%     The corners in 2D (Pts_W) map to the 3D RGB corners found above.
Pts_W = [-1, -1, 1;
         -1,  1, 1;
          1,  1, 1;
          1, -1, 1];                % (4×3), homogeneous 2D points

% Pts_RGB_corners' = M_2DWToRGB * Pts_W'
M_2DWToRGB = Pts_RGB_corners' * pinv(Pts_W');
M_RGBTo2DW = inv(M_2DWToRGB);

% Verify the 4 corner constraints are satisfied (loose tolerance for sampling)
if max(abs(M_2DWToRGB * Pts_W' - Pts_RGB_corners')) > 1e-4
    error('Corner fit check failed: 2D→RGB transform is inconsistent.');
end

% --- Build a visualization grid in the 2D plane and map to RGB for plotting
ngrid_1d_w     = 5;
ngrid_1d_w_bds = [-0.7, 0.7];
[s2, t2] = ndgrid(linspace(ngrid_1d_w_bds(1), ngrid_1d_w_bds(2), ngrid_1d_w), ...
                  linspace(ngrid_1d_w_bds(1), ngrid_1d_w_bds(2), ngrid_1d_w));
grid_Pts_2DW = [s2(:), t2(:)];                          % (K×2)
grid_Pts_3DW = [grid_Pts_2DW, ones(ngrid_1d_w^2, 1)];   % (K×3) 
grid_Pts_RGB = (M_2DWToRGB * grid_Pts_3DW')';           % (K×3)

% save('Pts_cc.mat', 'Pts_RGB', 'Pts_coneContrast', 'Pts_LMS', 'M_2DWToRGB', 'M_RGBTo2DW');

%% --- 3D visualization: LS plane vs. prior isoluminant plane
figure; hold on
% Filled polygon for the new LS plane (semi-transparent)
h1 = fill3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), ...
           [0.9, 0.9, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'k', 'LineWidth', 1.5);

% Colored samples on the plane (their own RGB as marker color)
for n = 1:ngrid_1d_w^2
    cmap = grid_Pts_RGB(n,:);
    scatter3(grid_Pts_RGB(n,1), grid_Pts_RGB(n,2), grid_Pts_RGB(n,3), ...
             100, 'filled', 'MarkerFaceColor', cmap);
end

% Boundary sampling trace and plane corners
plot3(Pts_RGB_inGamut(:,1), Pts_RGB_inGamut(:,2), Pts_RGB_inGamut(:,3), 'Color','k');
scatter3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), ...
         100, 'filled', 'MarkerFaceColor', 'k');

% Prior isoluminant plane outline (red)
scatter3(cornerPointsRGB_isoluminant(:,1), cornerPointsRGB_isoluminant(:,2), ...
         cornerPointsRGB_isoluminant(:,3), 50, 'filled', 'MarkerFaceColor', 'red');
h2 = fill3(cornerPointsRGB_isoluminant(:,1), cornerPointsRGB_isoluminant(:,2), ...
           cornerPointsRGB_isoluminant(:,3), [1, 0, 0], ...
           'FaceAlpha', 0, 'EdgeColor', 'red', 'LineWidth', 1.5);

xlabel('R'); ylabel('G'); zlabel('B');
axis equal tight; box on; grid on;
xlim([0 1]); ylim([0 1]); zlim([0 1]);
legend([h1, h2], {sprintf('%s cone isolating plane', isolating_plane), ...
                  'Isoluminant plane'});
title(sprintf('%s cone isolating plane', isolating_plane));
view(3)

%% --- 2D visualization in plane coordinates: compare LS vs isoluminant mapping
figure;
for m = 1:2
    subplot(1,2,m)
    for n = 1:ngrid_1d_w^2
        if m == 1
            % New LS plane mapping
            cmap = grid_Pts_RGB(n,:);
        else
            % Prior isoluminant mapping of the same 2D grid
            cmap = M_2DWToRGB_isoluminant * grid_Pts_3DW(n,:)';
        end
        scatter(grid_Pts_2DW(n,1), grid_Pts_2DW(n,2), 200, ...
                'filled', 'MarkerFaceColor', cmap); hold on
    end
    xlim([-1, 1]); ylim([-1, 1]);
    axis square; box on; grid on;
    xlabel('Model space dimension 1'); ylabel('Model space dimension 2');
    if m==1; ttl = sprintf('%s cone isolating plane', isolating_plane);
    else; ttl = 'Isoluminant plane';
    end
    title(ttl);
end

