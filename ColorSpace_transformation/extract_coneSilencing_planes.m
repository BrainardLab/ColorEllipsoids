clear all; close all; clc
% ------------------------------------------------------------------------
% 1) Load calibration and color-space transforms (from eLife setup)
% -------------------------------------------------------------------------
curr_dir = pwd;
S = load(fullfile(curr_dir, 'FilesFromPsychtoolbox', ...
                  'Transformation_btw_color_spaces.mat'), ...
                  'DELL_02242025_texture_right');

bg_LMS      = S.DELL_02242025_texture_right.bgLMS;         % 1x3 background LMS
M_RGBToLMS  = S.DELL_02242025_texture_right.M_RGBToLMS;    % 3x3 linear transform
M_LMSToRGB  = S.DELL_02242025_texture_right.M_LMSTORGB;

%% ------------------------------------------------------------------------
% 2) Sample directions on the unit sphere, scale to small RGB perturbations
%    about mid-gray [0.5,0.5,0.5], then convert to LMS and cone contrast
% -------------------------------------------------------------------------
n_azimuth = 5000;     % theta samples in [0, 2π)
n_polar   = 2000;     % phi samples in [0,  π]
rgb_radius = 0.1;     % step size in RGB around gray

% Unit sphere grid: size (n_polar x n_azimuth x 3)
sphere_xyz = UnitCircleGenerate_3D(n_azimuth, n_polar);

% Flatten to (n_polar*n_azimuth x 3) and scale to a radius in RGB
sphere_pts = reshape(sphere_xyz, [], 3) * rgb_radius;

% Center around monitor gray; transpose to 3xN for matrix ops
bg_rgb = [0.5, 0.5, 0.5];
rgb_samples = (sphere_pts + bg_rgb).';

% Convert to LMS, then to cone contrast (both 3xN)
lms_samples     = M_RGBToLMS * rgb_samples;
cone_contrast   = ExcitationToContrast(lms_samples, bg_LMS);

%% ------------------------------------------------------------------------
% 3) Find samples with (approximately) zero S-cone contrast (S ≈ 0)
% -------------------------------------------------------------------------
tol_S = 1e-7;
s0_idx = find(abs(cone_contrast(3,:)) < tol_S);

% Keep two extreme samples along the sampling order (bookends on S≈0 set)
cc_S0_endpoints = cone_contrast(:, [s0_idx(1), s0_idx(end)]).';

v1 = cc_S0_endpoints(1,:);   % 1x3
v2 = cc_S0_endpoints(2,:);   % 1x3

% 7x7 coefficient grid (covers [-1,1] along each spanning direction)
[s, t] = ndgrid(linspace(-10/3, 10/3, 3), linspace(-10/3, 10/3, 3));

% Points on the plane: P(s,t) = s*v1 + t*v2
Px = s * v1(1) + t * v2(1);   % 3x3
Py = s * v1(2) + t * v2(2);   % 3x3
Pz = s * v1(3) + t * v2(3);   % 3x3

% Build plane grid in CONE-CONTRAST space
P_cc = [Px(:), Py(:), Pz(:)]; % 9 x 3

% Convert cc → LMS → RGB for plotting in RGB axes
LMS_3d = ContrastToExcitation(P_cc.', bg_LMS);   % 3 x 9
Pts_RGB_all = (M_LMSToRGB * LMS_3d).';           % 9 x 3 

%sanity check
LMS_3d_sanity = M_RGBToLMS * Pts_RGB_all';                    
coneContrast_3d_sanity = ExcitationToContrast(LMS_3d_sanity, bg_LMS);
if min(abs(coneContrast_3d_sanity(3,:))) > 1e-4
    error('Sanity check did not pass.')
end

%% select and save
mask_bds = [0.15, 0.85];
mask = (Pts_RGB_all(:,1) > mask_bds(1) & Pts_RGB_all(:,1) < mask_bds(2)) & ...
       (Pts_RGB_all(:,2) > mask_bds(1) & Pts_RGB_all(:,2) < mask_bds(2));

idx = find(mask); 

Pts_RGB = Pts_RGB_all(idx, :);
Pts_coneContrast = coneContrast_3d_sanity(:,idx)';
Pts_LMS = LMS_3d_sanity(:,idx)';

Pts_RGB_mean = mean(Pts_RGB); 
[Pts_RGB_inGamut, Pts_RGB_corners] = planeCubeBoundarySample(Pts_RGB(1,:) - Pts_RGB_mean,...
                                    Pts_RGB(2,:) - Pts_RGB_mean, Pts_RGB_mean);

% save('Pts_RGB.mat', 'Pts_RGB', 'Pts_coneContrast', 'Pts_LMS', 'bg_LMS', 'M_RGBToLMS');


%%
figure; hold on
fill3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), [0.2 0.6 1.0], ...
      'FaceAlpha', 0.35, 'EdgeColor', 'k', 'LineWidth', 1.5);
scatter3(Pts_RGB(:,1), Pts_RGB(:,2), Pts_RGB(:,3), 20, 'filled', 'MarkerFaceColor', 'k'); 
scatter3(Pts_RGB_inGamut(:,1), Pts_RGB_inGamut(:,2), Pts_RGB_inGamut(:,3),2);
scatter3(Pts_RGB_corners(:,1), Pts_RGB_corners(:,2), Pts_RGB_corners(:,3), 100, 'filled', 'MarkerFaceColor', 'r');
xlabel('R'); ylabel('G'); zlabel('B');
axis equal tight; box on; grid on;
xlim([0 1]); ylim([0 1]); zlim([0 1]);
title('S-cone silencing plane');
view(3)
