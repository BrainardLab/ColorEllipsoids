%This script simulates isothreshold contours 
clear all; close all; clc
addpath(genpath('/Users/fangfang/Documents/MATLAB/toolboxes/gif/'))
addpath(genpath('/Users/fangfang/Documents/MATLAB/toolboxes/ellipsoid_fit/'))

%% load data from psychtoolbox
% Load in LMS cone fundamentals
S = [400 5 61];
load T_cones_ss2.mat
T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);
param.T_cones = T_cones; 
%size: 3 (cone types) x 61 (sampled wavelengths)
% save('T_cones.mat','T_cones');

% Load in primaries for a monitor
load B_monitor.mat
B_monitor = SplineSpd(S_monitor,B_monitor,S);
param.B_monitor = B_monitor;
%size: 61 (sampled wavelengths) x 3 (primaries)
% M_RGBToLMS = T_cones*B_monitor;
% save('B_monitor.mat','B_monitor');

% Load in XYZ color matching functions
load T_xyzCIEPhys2.mat
T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
M_LMSToXYZ = ((param.T_cones)'\(T_xyz)')';
param.M_LMSToXYZ = M_LMSToXYZ;
%T_xyz = (param.M_LMSToXYZ * param.T_cones)'
% save('M_LMSToXYZ.mat','M_LMSToXYZ');

%% Initialize reference stimuli within a RGB cube
% define a 5 x 5 x 5 grid of RGB values as reference stimuli
stim.nGridPts_ref = 5;
%define grid points from 0.2 to 0.8 in each dimension
stim.grid_ref     = linspace(0.2, 0.8, stim.nGridPts_ref);
%generate 3D grid
[stim.x_grid_ref, stim.y_grid_ref, stim.z_grid_ref] = ...
    ndgrid(stim.grid_ref,stim.grid_ref, stim.grid_ref);
% Concatenate grids to form reference points matrix
stim.ref_points = cat(4, stim.x_grid_ref, stim.y_grid_ref, stim.z_grid_ref);

%% compute iso-threshold contour
% Define a neutral background RGB value for the simulations
stim.background_RGB    = ones(3,1).*0.5;

% Define angular sampling for simulating chromatic directions on a sphere
% Sample 17 directions along the XY plane (azimuthal)
stim.numDirPts_xy = 17; 
% Sample directions along Z (polar), fewer due to spherical geometry
stim.numDirPts_z  = ceil(stim.numDirPts_xy/2);
% Azimuthal angle, 0 to 360 degrees
stim.grid_theta   = linspace(0, 2*pi,stim.numDirPts_xy);
% Polar angle, 0 to 180 degrees
stim.grid_phi     = linspace(0, pi, stim.numDirPts_z); 
% Create a grid of angles, excluding the redundant final theta
[stim.grid_THETA, stim.grid_PHI] = meshgrid(stim.grid_theta(1:end-1), stim.grid_phi);

% Calculate Cartesian coordinates for direction vectors on a unit sphere
stim.grid_x       = sin(stim.grid_PHI).*cos(stim.grid_THETA);
stim.grid_y       = sin(stim.grid_PHI).*sin(stim.grid_THETA);
stim.grid_z       = cos(stim.grid_PHI);
stim.grid_xyz     = cat(3, stim.grid_x, stim.grid_y, stim.grid_z);
% visualize chromatic directions in 3D
flag_visualize_chromDir = false;
if flag_visualize_chromDir
    figure; colormap(white)
    surf(horzcat(stim.grid_x, stim.grid_x(:,1)), ...
        horzcat(stim.grid_y,stim.grid_y(:,1)),...
        horzcat(stim.grid_z, stim.grid_z(:,1)),'FaceAlpha',0.3);
    xlabel('R'); ylabel('G'); zlabel('B');
    axis vis3d equal; camlight; lighting phong
    %save the figure
    analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
    myFigDir = 'Simulation_FigFiles';
    outputDir = fullfile(analysisDir, myFigDir);
    figFilePath = fullfile(outputDir, 'ChromaticDirs_3D.pdf');
    saveas(gcf, figFilePath);
end

%define threshold as deltaE = 1
stim.deltaE_1JND  = 1;

%the raw isothreshold contour is very tiny, we can amplify it by 5 times
%for the purpose of visualization
results.ellipsoid_scaler = 5;
%make a finer grid for the direction (just for the purpose of
%visualization)
plt.nThetaEllipsoid    = 200;
plt.nPhiEllipsoid      = 100;
plt.circleIn3D         = UnitCircleGenerate_3D(plt.nThetaEllipsoid, ...
    plt.nPhiEllipsoid);

%% Fitting starts from here
%for each reference stimulus
for i = 1:stim.nGridPts_ref
    disp(i)
    for j = 1:stim.nGridPts_ref
        for k = 1:stim.nGridPts_ref
            %grab the reference stimulus's RGB
            rgb_ref_ijk = squeeze(stim.ref_points(i,j,k,:));
            %convert it to Lab
            [ref_Lab_ijk, ~, ~] = convert_rgb_lab(param.B_monitor,...
                stim.background_RGB, param.T_cones, param.M_LMSToXYZ,... 
                rgb_ref_ijk);
            results.ref_Lab(i,j,k,:) = ref_Lab_ijk;
            
            %for each chromatic direction
            for l = 1:stim.numDirPts_z
                for m = 1:stim.numDirPts_xy-1
                    %determine the direction we are going 
                    vecDir = [stim.grid_x(l,m);...
                              stim.grid_y(l,m);...
                              stim.grid_z(l,m)];
        
                    %run fmincon to search for the magnitude of vector that
                    %leads to a pre-determined deltaE
                    results.opt_vecLen(i,j,k,l,m) = find_vecLen(...
                        stim.background_RGB, rgb_ref_ijk, ref_Lab_ijk, ...
                        vecDir, param,stim);
                end
            end
    
            %fit an ellipsoid
            [results.fitEllipsoid(i,j,k,:,:), ...
                results.fitEllipsoid_unscaled(i,j,k,:,:), ...
                results.rgb_surface_scaled(i,j,k,:,:,:),...
                results.rgb_surface_cov(i,j,k,:,:,:), ...
                results.ellipsoidParams{i,j,k}] = ...
                fit_3d_isothreshold_ellipsoid(rgb_ref_ijk, [],stim.grid_xyz, ...
                'vecLength',squeeze(results.opt_vecLen(i,j,k,:,:)),...
                'nThetaEllipsoid',plt.nThetaEllipsoid,...
                'nPhiEllipsoid',plt.nPhiEllipsoid,...
                'ellipsoid_scaler',results.ellipsoid_scaler);

            %if we want to find the ellipse that's the intersection between
            %the ellipsoid and a plane 
            % 1. GB with fixed R plane
            % 2. RB with fixed G plane
            % 3. RG with fixed B plane
            for n = 1:3 
                vecDir_oneFixedPlane = NaN(3,stim.numDirPts_xy-1);
                for o = 1:stim.numDirPts_xy-1 
                    %the chromatic direction should be
                    %[cos(theta)*sin(phi); sin(theta)*sin(phi); sin(phi)]
                    if n == 1 %fixed R plane
                        %phi goes from 0 to 2pi, theta = pi/2; cos(theta) = 0; sin(theta) = 1;
                        %in the following line, stim.grid_theta actually
                        %means stim.grid_phi but I don't want to create a
                        %new variable stim.grid_phi_allthewayto2pi
                        vecDir_oneFixedPlane(:,o) = [0; sin(stim.grid_theta(o));...
                            cos(stim.grid_theta(o))];
                    elseif n == 2 %fixed G plane
                        %phi goes from 0 to 2pi, theta = 0; cos(0) = 1; sin(0) = 0;
                        %in the following line, stim.grid_theta actually
                        %means stim.grid_phi but I don't want to create a
                        %new variable stim.grid_phi_allthewayto2pi
                        vecDir_oneFixedPlane(:,o) = [sin(stim.grid_theta(o));...
                            0; cos(stim.grid_theta(o))];
                    else %fixed B plane
                        %phi = pi/2; sin(phi) = 1; theta goes from 0 to 2pi
                        vecDir_oneFixedPlane(:,o) = [cos(stim.grid_theta(o));...
                            sin(stim.grid_theta(o));0];   
                    end
                    results.opt_vecLen_oneFixedPlane(i,j,k,n,o) = find_vecLen(...
                        stim.background_RGB, rgb_ref_ijk, ref_Lab_ijk, ...
                        vecDir_oneFixedPlane(:,o), param,stim);    
                end

                %fit an ellipse
                [results.fitEllipse(i,j,k,n,:,:), ...
                    results.fitEllipse_unscaled(i,j,k,n,:,:), ...
                    results.rgb_contour_scaled(i,j,k,n,:,:),...
                    results.rgb_contour_cov(i,j,k,n,:,:), ...
                    results.ellipseParams(i,j,k,n,:),...
                    results.AConstraint(i,j,k,n,:,:),...
                    results.Ainv(i,j,k,n,:,:), results.Q(i,j,k,n,:,:),~] = ...
                    fit_2d_isothreshold_contour(rgb_ref_ijk, [],...
                    vecDir_oneFixedPlane(setdiff(1:3,n),:), ...
                    'vecLength',squeeze(results.opt_vecLen_oneFixedPlane(i,j,k,n,:)),...
                    'varyingRGBplane',setdiff(1:3,n), ...
                    'nThetaEllipse',plt.nThetaEllipsoid,...
                    'ellipse_scaler',results.ellipsoid_scaler);
            end
        end
    end
end

%% visualize the iso-threshold contour
plot_3D_isothreshold_ellipsoid(stim.grid_ref, stim.grid_ref, ...
    stim.grid_ref,results.fitEllipsoid, plt.nThetaEllipsoid,...
    plt.nPhiEllipsoid,'slc_x_grid_ref',1:2:5,...
    'slc_y_grid_ref', 1:2:5, ...
    'slc_z_grid_ref',1:2:5,...
    'visualize_thresholdPoints',true,...
    'threshold_points', results.rgb_surface_scaled,...
    'color_ref_rgb', [0.2,0.2,0.2],...
    'color_surf',[0.8,0.8,0.8],...
    'color_threshold',[],...
    'saveFig',false,...
    'normalizedFigPos',[0,0.1,0.3,0.6]);

%% save the data
D = {param, stim, results, plt};
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'Simulation_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end
outputName = fullfile(outputDir, "Isothreshold_ellipsoid_CIELABderived.mat");
save(outputName,'D');


