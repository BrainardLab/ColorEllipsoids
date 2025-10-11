% -------------------------------------------------------------------------
% GOAL
% -------------------------------------------------------------------------
% Map a set of reference stimuli from *my* model space into *other observers’*
% model spaces that differ in cone fundamentals, via a common cone-contrast
% space.
%
% Pipeline (conceptually):
%   my model space (2D plane + filler dim)  -->  RGB (my calibration)
%   RGB (my calibration)                    -->  LMS (my cones)        --> cone contrasts (relative to my bg LMS)
%   cone contrasts                          -->  LMS (other observer)  --> RGB (other observer’s model space)
%
% We then visualize how the same “reference set” would appear in RGB for
% other observers with different cone fundamentals.
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
grid_RGB = M_2DWToRGB * grid_3d;

%% ------------------------------------------------------------------------
% 3) Visualize the reference set inside the unit RGB cube + the plane fill
% -------------------------------------------------------------------------
figure; hold on; box on; grid on

% Plane fill from the 4 corner points (convex hull gives two triangles)
V = corner_PointsRGB.';            % 4x3, rows are [R G B] vertices
K = convhull(V(:,1), V(:,2), V(:,3));

trisurf(K, V(:,1), V(:,2), V(:,3), ...
    'FaceColor', [0.5 0.5 0.5], 'FaceAlpha', 0.25, ...
    'EdgeColor', 'none');

% Plot the 49 reference RGBs (color each point by its own RGB)
R = grid_RGB(1,:); G = grid_RGB(2,:); B = grid_RGB(3,:);
scatter3(R, G, B, 50, [R.' G.' B.'], 'filled');

xlabel('R'); ylabel('G'); zlabel('B');
xlim([0 1]); ylim([0 1]); zlim([0 1]);
axis equal
view(35,25)

% Draw a wireframe unit cube for context
corners = [0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1];
edges   = [1 2; 2 3; 3 4; 4 1; 5 6; 6 7; 7 8; 8 5; 1 5; 2 6; 3 7; 4 8];
for k = 1:size(edges,1)
    plot3(corners(edges(k,[1 2]),1), corners(edges(k,[1 2]),2), corners(edges(k,[1 2]),3), ...
        'k-', 'LineWidth', 1.2);
end

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
T_xyz = 683*SplineCmf(S_xyzCIEPhys2, T_xyzCIEPhys2, Scolor);  
SetSensorColorSpace(calObjXYZ, T_xyz, Scolor);

% Load a bank of alternative cone fundamentals (cell array)
savedConeFundamentals = load([pwd, '/FilesFromPsychtoolbox/T_conesRnd.mat'], 'T_conesRnd');
nSets = length(savedConeFundamentals.T_conesRnd);

% Preallocate outputs across observers
grid_RGB_hypSub  = NaN(nSets, 3, nRefs);    % RGB for each alt observer (nSets x 3 x N)
M_RGBToLMS_hypSub = NaN(nSets, 3, 3);       % per-observer RGB->LMS
bgLMS_hypSub     = NaN(nSets, 3);           % per-observer bg LMS

for n = 1:nSets
    % -- 4a) Install the nth cone fundamentals into a fresh cal object
    T_cones_n = savedConeFundamentals.T_conesRnd{n};
    calObjCones = ObjectToHandleCalOrCalStruct(cal);

    % Gamma settings for device → linear primaries (assumes gammaMode exists)
    SetGammaMethod(calObjCones, gammaMode); 

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
    LMS_3d_hypSub = M_RGBToLMS_n * grid_RGB;                        % (3xN)
    coneContrast_3d = ExcitationToContrast(LMS_3d_hypSub, bgLMS_hypSub(n,:));

    % -- 4e) Map those contrasts back to absolute LMS using *my* bgLMS
    %        (This anchors cone-contrast definition to my background)
    LMS_3d = ContrastToExcitation(coneContrast_3d, bgLMS);

    % -- 4f) Finally, map LMS back to *my* RGB space for visualization
    grid_RGB_hypSub(n,:,:) = M_LMSToRGB * LMS_3d;

    clear calObjCones
end

%% ------------------------------------------------------------------------
% 5) Visualize each “other observer” mapping in RGB
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
save([pwd, '/FilesFromPsychtoolbox/grid_RGB_hypSub.mat'], "grid_RGB_hypSub");





