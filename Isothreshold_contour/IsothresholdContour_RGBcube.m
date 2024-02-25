%This script simulates isothreshold contours 
clear all; close all; clc
addpath(genpath('/Users/fangfang/Documents/MATLAB/toolboxes/gif/'))

%% load data from psychtoolbox
% Load in LMS cone fundamentals
S = [400 5 61];
load T_cones_ss2.mat
T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);
param.T_cones = T_cones; 
%size: 3 (cone types) x 61 (sampled wavelengths)
save('T_cones.mat','T_cones');

% Load in primaries for a monitor
load B_monitor.mat
B_monitor = SplineSpd(S_monitor,B_monitor,S);
param.B_monitor = B_monitor;
%size: 61 (sampled wavelengths) x 3 (primaries)
% M_RGBToLMS = T_cones*B_monitor;
save('B_monitor.mat','B_monitor');

% Load in XYZ color matching functions
load T_xyzCIEPhys2.mat
T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
M_LMSToXYZ = ((param.T_cones)'\(T_xyz)')';
param.M_LMSToXYZ = M_LMSToXYZ;
%T_xyz = (param.M_LMSToXYZ * param.T_cones)'
save('M_LMSToXYZ.mat','M_LMSToXYZ');

%% First create a cube and select the RG, the RB and the GB planes
%discretize RGB values
param.nGridPts               = 100;
param.grid                   = linspace(0, 1, param.nGridPts);
[param.x_grid, param.y_grid] = meshgrid(param.grid, param.grid);

%number of selected planes
param.nPlanes                = 3;
%for RG / RB / GB plane, we fix the B / G / R value to be one of the
%following
stim.fixed_RGBvec            = 0.1:0.1:0.9;
stim.len_fixed_RGBvec        = length(stim.fixed_RGBvec);

%get the grid points for those three planes with one dimension having a
%specific fixed value
for l = 1:stim.len_fixed_RGBvec
    param.plane_points{l} = get_gridPts(param.x_grid, param.y_grid,...
        1:param.nPlanes, stim.fixed_RGBvec(l).*ones(1, param.nPlanes));
end

%% set a grid for the reference stimulus
%pick 5 x 5 reference points 
stim.nGridPts_ref = 5;
stim.grid_ref     = linspace(0.2, 0.8, stim.nGridPts_ref);
% %pick 9 x 9 reference points 
% stim.nGridPts_ref = 9;
% stim.grid_ref     = linspace(0.1, 0.9, stim.nGridPts_ref);
[stim.x_grid_ref, stim.y_grid_ref] = meshgrid(stim.grid_ref,stim.grid_ref);

%get the grid points for the reference stimuli of each plane
for l = 1:stim.len_fixed_RGBvec
    stim.ref_points{l} = get_gridPts(stim.x_grid_ref, stim.y_grid_ref,...
        1:param.nPlanes, stim.fixed_RGBvec(l).*ones(1, param.nPlanes));
end

%% visualize the color planes
%select the slices we want to visualize 
plt.colormapMatrix = param.plane_points;
plot_3D_RGBplanes({param.plane_points{5}}, {plt.colormapMatrix{5}},...
    'ref_points', {stim.ref_points{5}}, 'paperSize',[40,20],'saveFig', false)

%visualize all slices
plot_3D_RGBplanes(param.plane_points, plt.colormapMatrix,...
    'ref_points', stim.ref_points,'paperSize',[40,20], 'saveFig', false)

%% compute iso-threshold contour
%set the background RGB
% stim.background_RGB    = ones(param.nPlanes, length(fixed_RGB_slc)).*0.5;
stim.background_RGB    = ones(param.nPlanes,1).*stim.fixed_RGBvec;

%sample total of 17 directions (0 to 360 deg) but the 1st and the last are the same
stim.numDirPts         = 17;
stim.grid_theta        = linspace(0, 2*pi,stim.numDirPts);
stim.grid_theta_xy     = [cos(stim.grid_theta(1:end-1)); ...
                          sin(stim.grid_theta(1:end-1))];
%define threshold as deltaE = 0.5
stim.deltaE_1JND       = 1;

%the raw isothreshold contour is very tiny, we can amplify it by 10 times
%for the purpose of visualization
results.contour_scaler = 5;
%make a finer grid for the direction (just for the purpose of
%visualization)
plt.nThetaEllipse      = 200;
plt.circleIn2D         = UnitCircleGenerate(plt.nThetaEllipse);

%%
%for each fixed R / G / B value in the BG / RB / RG plane
for l = 1:stim.len_fixed_RGBvec
    disp(l)
    %set the background RGB 
    background_RGB_l = stim.background_RGB(:,l);
    %for each plane
    for p = 1:param.nPlanes
        %vecDir is a vector that tells us how far we move along a specific direction 
        vecDir = NaN(param.nPlanes,1); vecDir(p) = 0; 
        %indices for the varying chromatic directions 
        %GB plane: [2,3]; RB plane: [1,3]; RG plane: [1,2]
        idx_varyingDim = setdiff(1:param.nPlanes,p);

        %for each reference stimulus
        for i = 1:stim.nGridPts_ref
            for j = 1:stim.nGridPts_ref
                %grab the reference stimulus's RGB
                rgb_ref_pij = squeeze(stim.ref_points{l}{p}(i,j,:));
                %convert it to Lab
                [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param.B_monitor,...
                    background_RGB_l, param.T_cones, param.M_LMSToXYZ,... 
                    rgb_ref_pij);
                results.ref_Lab(l,p,i,j,:) = ref_Lab_lpij;
                
                %for each chromatic direction
                for k = 1:stim.numDirPts-1
                    %determine the direction we are going 
                    vecDir(idx_varyingDim) = stim.grid_theta_xy(:,k);
    
                    %run fmincon to search for the magnitude of vector that
                    %leads to a pre-determined deltaE
                    results.opt_vecLen(l,p,i,j,k) = find_vecLen(...
                        background_RGB_l, rgb_ref_pij, ref_Lab_lpij, ...
                        vecDir, param,stim);
                end
                
                %fit an ellipse
                [results.fitEllipse(l,p,i,j,:,:), ...
                    results.fitEllipse_unscaled(l,p,i,j,:,:), ...
                    results.rgb_contour_scaled(l,p,i,j,:,:),...
                    results.rgb_contour_cov(l,p,i,j,:,:), ...
                    results.ellParams(l,p,i,j,:),...
                    results.AConstraint(l,p,i,j,:,:),...
                    results.Ainv(l,p,i,j,:,:), results.Q(l,p,i,j,:,:),~] = ...
                    fit_2d_isothreshold_contour(rgb_ref_pij, [],stim.grid_theta_xy, ...
                    'vecLength',squeeze(results.opt_vecLen(l,p,i,j,:)),...
                    'varyingRGBplane',idx_varyingDim, ...
                    'nThetaEllipse',plt.nThetaEllipse,...
                    'ellipse_scaler',results.contour_scaler);
            end
        end
    end
end

%% visualize the iso-threshold contour
plot_2D_isothreshold_contour(stim.x_grid_ref, stim.y_grid_ref, ...
    results.fitEllipse, stim.fixed_RGBvec,...
    'rgb_contour', results.rgb_contour_scaled,...
    'EllipsesLine','-',...
    'refColor',[1,1,1],...
    'visualizeRawData',true,...
    'saveFig',true,...
    'rgb_background',true)

%visualize just one slice
plot_2D_isothreshold_contour(stim.x_grid_ref, stim.y_grid_ref, ...
    results.fitEllipse(5,:,:,:,:,:), stim.fixed_RGBvec(5),...
    'rgb_contour', results.rgb_contour_scaled(5,:,:,:,:,:),...
    'refColor',[1,1,1],...
    'EllipsesLine','-',...
    'visualizeRawData',true,...
    'saveFig',true)

%% save the data
D = {param, stim, results, plt};
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'Simulation_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end
outputName = fullfile(outputDir, "Isothreshold_contour_CIELABderived.mat");
save(outputName,'D');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           HELPING FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [deltaE, comp_RGB] = compute_deltaE(vecLen, background_RGB, ...
    ref_RGB, ref_Lab, vecDir, param)
    %compute comparison RGB
    comp_RGB = ref_RGB + vecDir.*vecLen;

    %convert it to Lab
    [comp_Lab, ~, ~] = convert_rgb_lab(param.B_monitor,background_RGB,...
        param.T_cones, param.M_LMSToXYZ, comp_RGB);
    deltaE = norm(comp_Lab - ref_Lab);
end

function opt_vecLen = find_vecLen(background_RGB, ref_RGB, ref_Lab, ...
    vecDir, param, stim)
    deltaE = @(d) abs(compute_deltaE(d, background_RGB, ref_RGB,...
        ref_Lab, vecDir, param) - stim.deltaE_1JND);
    %have different initial points to avoid fmincon from getting stuck at
    %some places
    lb = 0; ub = 0.1;
    N_runs  = 1;
    init    = rand(1,N_runs).*(ub-lb) + lb;
    options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display','off');
    [vecLen_n, deltaE_n] = deal(NaN(1, N_runs));
    for n = 1:N_runs
        %use fmincon to search for the optimal defocus
        [vecLen_n(n), deltaE_n(n)] = fmincon(deltaE, init(n), ...
            [],[],[],[],lb,ub,[],options);
    end
    %find the index that corresponds to the minimum value
    [~,idx_min] = min(deltaE_n);
    %find the corresponding optimal focus that leads to the highest peak of
    %the psf's
    opt_vecLen = vecLen_n(idx_min);
end

