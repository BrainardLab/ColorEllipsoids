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
cal_date = '10062025';
cal_path = fullfile(getpref('ColorEllipsoids','ELPSMaterials'),'Calibration');
S = load(fullfile(cal_path, ['Transformation_btw_color_spaces_', cal_date, '.mat']), ...
                  ['DELL_',cal_date, '_texture_right']);
output_fig_dir = fullfile(cal_path, 'Plots', cal_date);
if ~isfolder(output_fig_dir)           
    [ok, msg] = mkdir(output_fig_dir);
    if ~ok; error('Could not create folder "%s": %s', output_fig_dir, msg); end
end

% Background/ambient and transforms (all 3x1 or 3x3 as noted)
bg_LMS      = S.DELL_10062025_texture_right.bgLMS;        % 1x3 background LMS (row)
ambient_LMS = S.DELL_10062025_texture_right.ambientLMS;   % 3x1 ambient LMS (col)
M_RGBToLMS  = S.DELL_10062025_texture_right.M_RGBToLMS;   % 3x3 RGB→LMS
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
%   • an origin, a L-only point, and S-only point.
% Any two independent vectors from these three will span the entire LS (2D) plane.
P_cc = [0,   0,   0;    % origin (bg)
        0.15, 0,   0;   % +L,  M=0, S=0
        0,   0, 0.15];  % +S,  M=0

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
mask_bds = [0, 1];
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
[inGamt_Pts_RGB, corner_Pts_RGB, e1, e2] = planeCubeBoundarySample( ...
    Pts_RGB(2,:)  - Pts_RGB_mean, ...   % v1 in RGB plane
    Pts_RGB(3,:) - Pts_RGB_mean, ...  % v2 in RGB plane (distinct from v1)
    Pts_RGB_mean, ...                   % plane origin in RGB
    'intersectionDims', isolating_dims);

% --- Solve for 2D→RGB transform using 4 labeled plane points (±1,±1)
%     We treat plane coordinates as [w1, w2, 1] with w ∈ {−1, +1}.
%     The corners in 2D (Pts_W) map to the 3D RGB corners found above.
corner_Pts_2DW = [-1, -1, 1;
                  -1,  1, 1;
                   1,  1, 1;
                   1, -1, 1];                % (4×3), homogeneous 2D points

% Pts_RGB_corners' = M_2DWToRGB * Pts_W'
M_2DWToRGB = corner_Pts_RGB' * pinv(corner_Pts_2DW');
M_RGBTo2DW = inv(M_2DWToRGB);

% Verify the 4 corner constraints are satisfied (loose tolerance for sampling)
if max(abs(M_2DWToRGB * corner_Pts_2DW' - corner_Pts_RGB')) > 1e-4
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
grid_Pts_LMS = M_RGBToLMS * grid_Pts_RGB' + ambient_LMS;
grid_Pts_cc = ExcitationToContrast(grid_Pts_LMS, bg_LMS);

%also compute the corresponding corner points
corner_Pts_LMS = M_RGBToLMS * corner_Pts_RGB' + ambient_LMS;
corner_Pts_cc = ExcitationToContrast(corner_Pts_LMS, bg_LMS);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure A: cone contrast plane
figure; ax = gca; hold(ax, 'on');

% --- Scatter all points at once (per-point RGB colors)
X = grid_Pts_cc(isolating_dims(1),:);
Y = grid_Pts_cc(isolating_dims(2),:);
scatter(X, Y, 100, grid_Pts_RGB, 'filled', 'MarkerEdgeColor','none');

% --- Two arrows from the origin
v1 = P_cc(2,  isolating_dims);    % [vx vy]
v2 = P_cc(end,isolating_dims);
quiver(0,0, v1(1), v1(2), 'LineWidth',2, 'MaxHeadSize',0.5, 'Color','k');
quiver(0,0, v2(1), v2(2), 'LineWidth',2, 'MaxHeadSize',0.5, 'Color','k');

% --- Corner points and closed polygon
C2 = corner_Pts_cc(isolating_dims,:);             % 2 x 4
scatter(C2(1,:), C2(2,:), 80, 'k', 'filled');
idx = [1:size(C2,2) 1];                           % wrap to close
plot(C2(1,idx), C2(2,idx), 'k-');

% --- Limits, ticks, labels
xmin = min(C2(1,:)); xmax = max(C2(1,:));
ymin = min(C2(2,:)); ymax = max(C2(2,:));
axis(ax,'equal'); 
padX = 0.10;              % x padding
padY = 0.10;              % y padding
xlim([xmin xmax] + [-padX padX]);
ylim([ymin ymax] + [-padY padY]);
xticks(linspace(round(xmin,1), round(xmax,1), 3));
yticks(linspace(round(ymin,1), round(ymax,1), 5));

box(ax,'on'); grid(ax,'on');
xlabel('L-cone contrast'); ylabel('M-cone contrast');
title('Cone contrast space');
set(ax,'FontSize',14);
outfile = fullfile(output_fig_dir, 'LSisolating_cc.pdf');  % target PDF
exportgraphics(gcf, outfile, 'ContentType','vector');      

%% 
% figure B: cone excitation plane
figure; ax = gca; hold(ax,'on');

% --- Data in the selected dims
X  = grid_Pts_LMS(isolating_dims(1),:);
Y  = grid_Pts_LMS(isolating_dims(2),:);
C2 = corner_Pts_LMS(isolating_dims,:);              % 2 x 4 (corners)
o  = Pts_LMS(1, isolating_dims);                    % origin in LMS plane [L S]
v1 = Pts_LMS(2, isolating_dims) - o;                % arrow 1 vector
v2 = Pts_LMS(3, isolating_dims) - o;                % arrow 2 vector

% --- Scatter all points at once (per-point RGB colors)
scatter(X, Y, 100, grid_Pts_RGB, 'filled', 'MarkerEdgeColor','none');

% --- Two arrows from the same origin
qargs = {'LineWidth',2,'MaxHeadSize',0.5,'Color','k'};
quiver(o(1), o(2), v1(1), v1(2), 0, qargs{:});
quiver(o(1), o(2), v2(1), v2(2), 0, qargs{:});

% --- Corner points and closed polygon
scatter(C2(1,:), C2(2,:), 80, 'k', 'filled');
plot(C2(1,[1:end 1]), C2(2,[1:end 1]), 'k-');

% --- Limits and ticks
xmin = min(C2(1,:)); xmax = max(C2(1,:));
ymin = min(C2(2,:)); ymax = max(C2(2,:));
axis(ax,'equal'); 
padX = 0.01; padY = 0.01;
xlim([xmin xmax] + [-padX padX]);
ylim([ymin ymax] + [-padY padY]);

xticks(linspace(round(xmin,1), round(xmax,1), 5));
yticks(linspace(round(ymin,1), round(ymax,1), 9));

box(ax,'on'); grid(ax,'on');
xlabel('L-cone excitation'); ylabel('S-cone excitation');
title('Cone excitation space');
set(ax,'FontSize',14);
outfile = fullfile(output_fig_dir, 'LSisolating_ce.pdf');  % target PDF
exportgraphics(gcf, outfile, 'ContentType','vector');      

%% 
% figure C: linear RGB
figure; ax = gca; hold(ax,'on');

% New LS plane (semi-transparent polygon)
h1 = fill3(corner_Pts_RGB(:,1), corner_Pts_RGB(:,2), corner_Pts_RGB(:,3), ...
           [0.9 0.9 0.9], 'FaceAlpha',0.35, 'EdgeColor','k', 'LineWidth',1.5);

% Colored samples on the plane (per-point RGB color)
scatter3(grid_Pts_RGB(:,1), grid_Pts_RGB(:,2), grid_Pts_RGB(:,3), ...
         100, grid_Pts_RGB, 'filled', 'MarkerEdgeColor','none');

% Boundary trace and plane corners
plot3(inGamt_Pts_RGB(:,1), inGamt_Pts_RGB(:,2), inGamt_Pts_RGB(:,3), 'k-');
scatter3(corner_Pts_RGB(:,1), corner_Pts_RGB(:,2), corner_Pts_RGB(:,3), ...
         100, 'k', 'filled');

% Prior isoluminant plane outline (red)
scatter3(cornerPointsRGB_isoluminant(:,1), cornerPointsRGB_isoluminant(:,2), ...
         cornerPointsRGB_isoluminant(:,3), 50, 'red', 'filled');
h2 = fill3(cornerPointsRGB_isoluminant(:,1), cornerPointsRGB_isoluminant(:,2), ...
           cornerPointsRGB_isoluminant(:,3), [1 0 0], ...
           'FaceAlpha',0, 'EdgeColor','red', 'LineWidth',1.5);

% Two basis arrows from the same origin (combine in one quiver3 call)
o = Pts_RGB(1,:);                     % origin
E = [e1(:) e2(:)];                    % 3x2 basis (columns = e1,e2)
quiver3(o(1)*[1 1], o(2)*[1 1], o(3)*[1 1], ...
        E(1,:),   E(2,:),   E(3,:), 0, ...
        'LineWidth',2, 'MaxHeadSize',0.5, 'Color','k');

% Axes styling
xlabel('R'); ylabel('G'); zlabel('B');
axis(ax,'equal'); axis(ax,'tight'); box(ax,'on'); grid(ax,'on');
set(ax, 'XLim',[0 1], 'YLim',[0 1], 'ZLim',[0 1], ...
        'XTick',0:0.25:1, 'YTick',0:0.25:1, 'ZTick',0:0.25:1, ...
        'FontSize',12);

legend([h1 h2], {sprintf('%s cone isolating plane', isolating_plane), ...
                 'Isoluminant plane'});
title('RGB space');
view(ax,3);
outfile = fullfile(output_fig_dir, 'LSisolating_RGB.pdf');  % target PDF
exportgraphics(gcf, outfile, 'ContentType','vector');      

%% 
% figure D: model space (2D)
% Precompute colors for each panel (still plotting in the inner loop)
C_ls  = grid_Pts_RGB;                               % panel 1 colors (LS plane)
C_iso = (M_2DWToRGB_isoluminant * grid_Pts_3DW')';  % panel 2 colors
C_all = {C_ls, C_iso};
T_all = {sprintf('%s cone isolating plane', isolating_plane), 'Isoluminant plane'};

X = grid_Pts_2DW(:,1);
Y = grid_Pts_2DW(:,2);
K = ngrid_1d_w^2;

figure;
for m = 1:2
    subplot(1,2,m); hold on

    % point-by-point scatter with panel-specific colors
    for n = 1:K
        c = C_all{m}(n,:);  % 1x3 RGB
        scatter(X(n), Y(n), 125, 'filled', 'MarkerFaceColor', c, 'MarkerEdgeColor','none');
    end

    % corners + polygon
    scatter(corner_Pts_2DW(:,1), corner_Pts_2DW(:,2), 100, 'filled', 'MarkerFaceColor','k');
    plot([corner_Pts_2DW(:,1); corner_Pts_2DW(1,1)], ...
         [corner_Pts_2DW(:,2); corner_Pts_2DW(1,2)], 'k-');

    % axes, ticks, labels
    xlim([-1.1, 1.1]); ylim([-1.1, 1.1]);
    xticks(linspace(-0.7, 0.7, 5)); yticks(linspace(-0.7, 0.7, 5));
    axis square; box on; grid on;
    xlabel('Model space dimension 1'); ylabel('Model space dimension 2');
    title(T_all{m});
    set(gca,'FontSize',12)
end

set(gcf, 'Units','normalized', 'Position',[0 0 0.5 0.4]);
set(gcf, 'PaperUnits','inches', 'PaperSize',[20 11]);   % for export
outfile = fullfile(output_fig_dir, 'LSisolating_2DW.pdf');  % target PDF
exportgraphics(gcf, outfile, 'ContentType','vector');     

%%
% save('Pts_cc.mat', 'Pts_RGB', 'Pts_coneContrast', 'Pts_LMS', 'M_2DWToRGB', 'M_RGBTo2DW');

