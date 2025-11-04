clear all; close all; clc
% -------------------------------------------------------------------------
% Goal
%   Construct an LM cone–isolating plane. We want to sample some stimuli
%   from this plane so that Nicolas can make isetbio simulations based on
%   his mRGC model.
% Steps
%   1) Choose points on the LM cone-contrast plane (S=0).
%   2) Map those points to LMS, then to RGB.
%   3) In RGB space, find the intersection polygon (plane ∩ [0,1]^3).
%   4) From that polygon’s four “corners”, derive a 2D↔RGB linear transform.
% -------------------------------------------------------------------------

% --- Load calibration & transforms (eLife setup) --------------------------
curr_dir = pwd;
S = load(fullfile(curr_dir, 'FilesFromPsychtoolbox', ...
                  'Transformation_btw_color_spaces.mat'), ...
                  'DELL_02242025_texture_right');

bg_LMS      = S.DELL_02242025_texture_right.bgLMS;       % 3x1 background LMS
ambient_LMS = S.DELL_02242025_texture_right.ambientLMS;  % 3x1 ambient LMS
M_RGBToLMS  = S.DELL_02242025_texture_right.M_RGBToLMS;  % 3x3 RGB→LMS
M_LMSToRGB  = S.DELL_02242025_texture_right.M_LMSTORGB;  % 3x3 LMS→RGB

% --- Hand-picked sample points on LM cone-contrast plane (S=0) ------------
% s,t specify LM coordinates in cone-contrast units; S coordinate is zero.
s = [-0.3, -0.15, 0, 0.15, 0.3, -0.05, 0.05]';
t = [-0.3, -0.15, 0, 0.15, 0.3,  0.05,-0.05]';
nSlc_pts = length(s);

% Define the isolating plane metadata 
isolating_plane = 'LM';
isolating_dims  = [1,2];   % indices for L and M
silencing_dim   = 3;       % index for S
P_cc = [s(:), t(:), zeros(length(s(:)),1)];   % N×3 LM(S=0) cone-contrast pts

% --- cc → LMS → RGB -------------------------------------------------------
LMS_3d  = ContrastToExcitation(P_cc', bg_LMS);      % 3×N: add background, go to LMS
% note, because the actual LMS cone excitations include contributions from
% ambient light, we subtract it to isolate portion due to the RGB stimulus
Pts_RGB = (M_LMSToRGB * (LMS_3d - ambient_LMS)).';  % N×3: LMS→RGB (device space)

% --- Sanity check: mapped points remain S-cone isolating (S-contrast ≈ 0) -
Pts_LMS = M_RGBToLMS * Pts_RGB' + ambient_LMS;                  % 3×N: RGB→LMS
Pts_coneContrast = (ExcitationToContrast(Pts_LMS, bg_LMS))';    % 3×N: back to cc

if min(abs(Pts_coneContrast(:,silencing_dim))) > 1e-4
    error('Sanity check failed: S-cone contrast not near zero.');
end

%% --- Compute RGB-gamut boundary intersection for the LM plane ------------
% Center a spanning pair at the plane’s mean to define in-plane directions.
Pts_RGB_mean = mean(Pts_RGB);
[Pts_RGB_inGamut, Pts_RGB_corners] = planeCubeBoundarySample( ...
    Pts_RGB(1,:) - Pts_RGB(3,:), ...   % dir 1 in-plane
    Pts_RGB(end,:) - Pts_RGB(3,:), ... % dir 2 in-plane
    Pts_RGB(3,:), ...                  % plane passes through this point
    'intersectionDims', isolating_dims); % (custom option in your function)

% --- Build 2D↔RGB linear transforms from four reference corners -----------
% By convention, map 2D W-plane corners ([-1,1] square at Wz=1) → RGB corners.
%   Pts_RGB_corners' = M_2DWToRGB * Pts_W'
Pts_W       = [-1, -1, 1;  -1, 1, 1;  1, 1, 1;  1, -1, 1];  % 4×3 (homog 2D)
M_2DWToRGB  = Pts_RGB_corners' * pinv(Pts_W');              % 3×3
M_RGBTo2DW  = inv(M_2DWToRGB);                              % 3×3 (for inverse map)

% --- Sanity check: the forward map reproduces the RGB corners -------------
if max(abs(M_2DWToRGB * Pts_W' - Pts_RGB_corners')) > 1e-4
    error('Sanity check failed: 2D↔RGB corner mapping is inconsistent.');
end

% save('Pts_cc.mat', 'Pts_RGB', 'Pts_coneContrast', 'Pts_LMS', 'bg_LMS', 'ambient_LMS', ...
%     'M_RGBToLMS', 'M_LMSToRGB', 'M_2DWToRGB', 'M_RGBTo2DW', 'Pts_RGB_inGamut', 'Pts_RGB_corners');

%% --- Plot: RGB-space polygon (plane ∩ cube) + sample points --------------
figure; hold on
fill3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), ...
      [0.9, 0.9, 0.9], 'FaceAlpha', 0.35, 'EdgeColor', 'k', 'LineWidth', 1.5);
for n = 1:nSlc_pts
    cmap = Pts_RGB(n,:);  % use actual RGB as the marker color
    scatter3(Pts_RGB(n,1), Pts_RGB(n,2), Pts_RGB(n,3), 100, ...
             'filled', 'MarkerFaceColor', cmap);
end
plot3(Pts_RGB_inGamut(:,1), Pts_RGB_inGamut(:,2), Pts_RGB_inGamut(:,3), 'Color','k');
scatter3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), ...
         100, 'filled', 'MarkerFaceColor', 'k');
xlabel('R'); ylabel('G'); zlabel('B');
axis equal tight; box on; grid on;
xlim([0 1]); ylim([0 1]); zlim([0 1]);
title(sprintf('%s cone–isolating plane (RGB space)', isolating_plane));
view(3)

% --- Plot: the same points in LM cone-contrast coordinates ----------------
figure; hold on
for n = 1:nSlc_pts
    cmap = Pts_RGB(n,:);
    scatter(Pts_coneContrast(n, isolating_dims(1)), ...
            Pts_coneContrast(n, isolating_dims(2)), ...
            200, 'filled', 'MarkerFaceColor', cmap);
end
xlabel(sprintf('%s cone contrast', isolating_plane(1)));   % 'L cone contrast'
ylabel(sprintf('%s cone contrast', isolating_plane(2)));   % 'M cone contrast'
xlim([min(Pts_coneContrast(:)), max(Pts_coneContrast(:))]);
ylim([min(Pts_coneContrast(:)), max(Pts_coneContrast(:))]); 
axis square; grid on
xticks(linspace(min(Pts_coneContrast(:)), max(Pts_coneContrast(:)), 5));
yticks(linspace(min(Pts_coneContrast(:)), max(Pts_coneContrast(:)), 5));

