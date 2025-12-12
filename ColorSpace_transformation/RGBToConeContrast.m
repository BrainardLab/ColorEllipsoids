% -------------------------------------------------------------------------
% GOAL
% -------------------------------------------------------------------------
% Map a fixed set of reference stimuli from an *other observer’s gamut space*
% into *my gamut space*, accounting for differences in cone fundamentals via
% a shared cone-contrast representation.
%
% We begin by defining an identical set of reference points in the *other
% observer’s gamut space*. These points are then transformed through the
% following conceptual pipeline:
%
%   other observer’s gamut space
%       → LMS excitations (using their cone fundamentals)
%       → cone contrast (relative to their background LMS)
%       → LMS excitations (using my cone fundamentals)
%       → my gamut space
%
% This procedure allows us to visualize how the same nominal set of reference
% stimuli would be represented in *my gamut space* for observers with
% different cone fundamentals.
%
% Notes:
% - “LMS excitations” = absolute cone responses. “Cone contrast” = (LMS - bgLMS) ./ bgLMS
% - All RGBs here are device-linear in [0,1].

clear all; close all; clc

%% ------------------------------------------------------------------------
% 1) Load calibration & transforms used in the eLife paper
% -------------------------------------------------------------------------
currentDir = pwd;
savedTransformations = load([currentDir,...
    '/FilesFromPsychtoolbox/Transformation_btw_color_spaces.mat'], ...
    'DELL_02242025_texture_right');

% Isoluminant plane corner points (3x4, columns are [R;G;B] corners)
corner_PointsRGB = savedTransformations.DELL_02242025_texture_right.corner_PointsRGB;

% Background LMS excitations for *my* observer/calibration (1x3)
bgLMS = savedTransformations.DELL_02242025_texture_right.bgLMS;

% Linear transforms between model (2D+filler) and RGB, and between RGB and LMS
M_2DWToRGB = savedTransformations.DELL_02242025_texture_right.M_2DWToRGB;  % (3x3) maps [d1; d2; 1] -> RGB
M_RGBToLMS = savedTransformations.DELL_02242025_texture_right.M_RGBToLMS;  % (3x3) maps RGB -> LMS
M_LMSToRGB = inv(M_RGBToLMS);                                              % inverse mapping LMS -> RGB (my calibration)

%% ------------------------------------------------------------------------
% 2) Define a 2D grid of reference stimuli on the isoluminant plane
% -------------------------------------------------------------------------
grid_1d = linspace(-0.7, 0.7, 7);           % 7x7 grid in model space
[grid_dim1, grid_dim2] = meshgrid(grid_1d, grid_1d);
nRefs = numel(grid_dim1);                    % 49 reference points

% Stack into 2D coordinates and append a filler dim of ones → 3xN (d1,d2,1)'
grid_2d = [grid_dim1(:), grid_dim2(:)];
grid_3d = [grid_2d, ones(nRefs, 1)]';

% Map 2D model points into RGB (3xN)
% the set of reference points in *other observer's gamut space*
% they stay the same
grid_RGB = M_2DWToRGB * grid_3d;  

%sanity check
LMS_3d_sanitycheck = M_RGBToLMS * grid_RGB; 
coneContrast_3d_sanitycheck = ExcitationToContrast(LMS_3d_sanitycheck, bgLMS);
LMS_3d_sanitycheck_b = ContrastToExcitation(coneContrast_3d_sanitycheck, bgLMS);
grid_RGB_sanitycheck = M_LMSToRGB * LMS_3d_sanitycheck_b;
if max(abs(grid_RGB_sanitycheck - grid_RGB)) > 1e-6
    disp('Sanity check did not pass.')
end

%% ------------------------------------------------------------------------
% 3) Visualize the reference set inside the unit RGB cube + the plane fill
% -------------------------------------------------------------------------
R = grid_RGB(1,:); G = grid_RGB(2,:); B = grid_RGB(3,:);
C = [R.' G.' B.'];

plot3d_points_with_plane(grid_RGB, ...
    'PlaneCorners', corner_PointsRGB, ...
    'PointColors', C, ...
    'Labels', {'R','G','B'}, ...
    'Marker', 'o',... 
    'Limits', [0 1; 0 1; 0 1]);

% visualize LMS
V2 = M_RGBToLMS * corner_PointsRGB;  
plot3d_points_with_plane(LMS_3d_sanitycheck, ...
    'PlaneCorners', V2, ...
    'PointColors', C, ...
    'Labels', {'L','M','S'}, ...
    'Marker', 'o');


% visualize cone contrast
V3 = ExcitationToContrast((M_RGBToLMS * corner_PointsRGB), bgLMS); 
plot3d_points_with_plane(coneContrast_3d_sanitycheck, ...
    'PlaneCorners', V3, ...
    'PointColors', C, ...
    'Labels', {'$\Delta L / L_{bg}$', '$\Delta M / M_{bg}$', '$\Delta S / S_{bg}$'},...
    'LabelInterpreter', 'latex', ...
    'Marker', 'o');

%% ------------------------------------------------------------------------
% 4) Prepare “other observers” by swapping cone fundamentals
%    and recomputing LMS/RGB under their cones and my device
% -------------------------------------------------------------------------
% Load the calibration object used to map between device and sensors
cal = savedTransformations.DELL_02242025_texture_right.cal;
calObjXYZ = ObjectToHandleCalOrCalStruct(cal);

% Set color matching functions (XYZ) for this calibration object (phys units)
% (Order here: we define T_xyz then set into cal object.)
load T_xyzCIEPhys2.mat

% Load a bank of alternative cone fundamentals (cell array)
savedConeFundamentals = load([pwd, '/FilesFromPsychtoolbox/T_conesRnd.mat'], 'T_conesRnd');
nSets = length(savedConeFundamentals.T_conesRnd);

% Preallocate outputs across observers
grid_RGB_hypSub  = NaN(nSets, 3, nRefs);    % RGB for each alt observer (nSets x 3 x N)
M_RGBToLMS_hypSub = NaN(nSets, 3, 3);       % per-observer RGB->LMS
bgLMS_hypSub     = NaN(nSets, 3);           % per-observer bg LMS

for n = 1:nSets %the last one includes the original cone fundamentals
    % -- 4a) Install the nth cone fundamentals into a fresh cal object
    T_cones_n = savedConeFundamentals.T_conesRnd{n};
    calObjCones = ObjectToHandleCalOrCalStruct(cal);

    % Gamma settings for device → linear primaries (assumes gammaMode exists)
    SetGammaMethod(calObjCones, 2); 

    % Wavelength support for sensors
    Scolor = calObjCones.get('S');
    SetSensorColorSpace(calObjCones, T_cones_n, Scolor);

    % -- 4b) Compute bg LMS for this observer 
    bgPrimary = SettingsToPrimary(calObjCones, PrimaryToSettings(calObjCones, [0.5 0.5 0.5]'));
    bgLMS_hypSub(n,:) = PrimaryToSensor(calObjCones, bgPrimary);

    % -- 4c) Build per-observer linear transforms
    M_RGBToLMS_n = T_cones_n * calObjXYZ.cal.P_device;
    M_RGBToLMS_hypSub(n,:,:) = M_RGBToLMS_n;

    % -- 4d) Take *my* RGB grid → per-observer LMS → *my* bg-referenced cone contrasts
    LMS_3d_hypSub = M_RGBToLMS_n * grid_RGB;                        
    coneContrast_3d = ExcitationToContrast(LMS_3d_hypSub, bgLMS_hypSub(n,:)'); %bgLMS has to be a column vector

    % -- 4e) Map those contrasts back to absolute LMS using *my* bgLMS
    %        (This anchors cone-contrast definition to my background)
    LMS_3d = ContrastToExcitation(coneContrast_3d, bgLMS);

    % -- 4f) Finally, map LMS back to *my* RGB space for visualization
    grid_RGB_hypSub(n,:,:) = M_LMSToRGB * LMS_3d;

    % transformation matrix that goes from RGB to the model space

    clear calObjCones
end

%% ------------------------------------------------------------------------
% 5) Visualize the same physical reference points in *my gamut space*
% -------------------------------------------------------------------------
figure; hold on; box on; grid on
for n = 1:nSets
    % (Re-)draw the isoluminant plane for reference
    trisurf(K, V(:,1), V(:,2), V(:,3), ...
        'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.25, ...
        'EdgeColor', 'none');

    % Extract this observer’s transformed RGB grid and plot
    R_n = squeeze(grid_RGB_hypSub(n, 1,:));
    G_n = squeeze(grid_RGB_hypSub(n, 2,:));
    B_n = squeeze(grid_RGB_hypSub(n, 3,:));
    scatter3(R_n', G_n', B_n', 50, [R_n, G_n, B_n], 'filled');

    % Unit cube overlay
    for k = 1:size(edges,1)
        plot3(corners(edges(k,[1 2]),1), corners(edges(k,[1 2]),2), ...
            corners(edges(k,[1 2]),3), 'k-', 'LineWidth', 1.2);
    end
    xlabel('R'); ylabel('G'); zlabel('B');
    xlim([0 1]); ylim([0 1]); zlim([0 1]);
    axis equal
    view(35,25)
    pause(1)
end

%% save data
%save([pwd, '/FilesFromPsychtoolbox/grid_RGB_hypSub_age50.mat'], "grid_RGB_hypSub");





