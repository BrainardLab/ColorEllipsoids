%This script simulates isothreshold contours 
clear all; close all; clc
% addpath(genpath('/Users/fangfang/Documents/MATLAB/toolboxes/gif/'))

%% load data from psychtoolbox
% Load in LMS cone fundamentals
S = [400 5 61];
load T_cones_ss2.mat
param.T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S); 
%size: 3 (cone types) x 61 (sampled wavelengths)

% Load in primaries for a monitor
load B_monitor.mat
param.B_monitor = SplineSpd(S_monitor,B_monitor,S);
%size: 61 (sampled wavelengths) x 3 (primaries)
% M_RGBToLMS = T_cones*B_monitor;

% Load in XYZ color matching functions
load T_xyzCIEPhys2.mat
T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
param.M_LMSToXYZ = ((param.T_cones)'\(T_xyz)')';
%T_xyz = (param.M_LMSToXYZ * param.T_cones)'

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
[stim.x_grid_ref, stim.y_grid_ref] = meshgrid(stim.grid_ref,stim.grid_ref);

%get the grid points for the reference stimuli of each plane
for l = 1:stim.len_fixed_RGBvec
    stim.ref_points{l} = get_gridPts(stim.x_grid_ref, stim.y_grid_ref,...
        1:param.nPlanes, stim.fixed_RGBvec(l).*ones(1, param.nPlanes));
end

%% visualize the color planes
%select the slices we want to visualize 
fixed_RGB_slc = 0.1:0.1:0.9;
plt.idx_fixed_RGB_slc = arrayfun(@(idx) find(stim.fixed_RGBvec == ...
    fixed_RGB_slc(idx)), 1:length(fixed_RGB_slc));
plt.ttl = {'GB plane', 'RB plane', 'RG plane'};
plt.colormapMatrix = param.plane_points;
plt.flag_save = true;

%visualize
% plot_3D_RGBplanes(param, stim, plt)

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
stim.deltaE_1JND       = 0.5;

%the raw isothreshold contour is very tiny, we can amplify it by 10 times
%for the purpose of visualization
results.contour_scaler = 10;
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
                [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param,...
                    background_RGB_l, rgb_ref_pij);
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

                % compute the iso-threshold contour 
                rgb_contour_lpij = rgb_ref_pij(idx_varyingDim)' + ...
                    repmat(squeeze(results.opt_vecLen(l,p,i,j,:).*...
                    results.contour_scaler), [1,2]).*stim.grid_theta_xy';
                results.rgb_contour(l,p,i,j,:,:) = rgb_contour_lpij;

                %fit an ellipse
                [~,~,~,fitQ_lpij] = FitEllipseQ(rgb_contour_lpij' - ...
                    rgb_ref_pij(idx_varyingDim),'lockAngleAt0',false);
                results.fitQ(l,p,i,j,:,:) = fitQ_lpij;
                results.fitEllipse(l,p,i,j,:,:) = (PointsOnEllipseQ(...
                    fitQ_lpij,plt.circleIn2D) + rgb_ref_pij(idx_varyingDim))';
            end
        end
    end
end

%% visualize the iso-threshold contour
plt.flag_visualizeRawData = true;
plot_isothreshold_contour(param, stim, results, plt)

%% save the data
D = {param, stim, results, plt};
save("Isothreshold_contour_CIELABderived.mat","D");

%% see if there exists analytical solutions
slc_fixedVal = 5;
lab_temp = cell(1,param.nPlanes);
for p = 1:param.nPlanes
    for i = 1:param.nGridPts
        for j = 1:param.nGridPts
            lab_temp{p}(i,j,:) = convert_rgb_lab(param,...
                squeeze(stim.background_RGB(:,slc_fixedVal)),...
                squeeze(param.plane_points{slc_fixedVal}{p}(i,j,:)));
        end
    end
end

figure
for p = 1:param.nPlanes
    subplot(1,param.nPlanes,p)
    surf(lab_temp{p}(:,:,1), lab_temp{p}(:,:,2), lab_temp{p}(:,:,3),...
        plt.colormapMatrix{slc_fixedVal}{p},'EdgeColor','none')
    axis square; xlabel('L'); ylabel('a'); zlabel('b');
end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           HELPING FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grid_pts = get_gridPts(X, Y, fixed_dim, val_fixed_dim)
    XY = {X,Y};
    grid_pts = cell(1,length(fixed_dim));
    for i = 1:length(fixed_dim)
        varying_dim = setdiff(1:3, fixed_dim(i));
        grid_pts_i = cell(1,3);
        grid_pts_i{fixed_dim(i)} = val_fixed_dim(i).*ones(size(X));
        for j = 1:length(varying_dim)
            grid_pts_i{varying_dim(j)} = XY{j};
        end
        grid_pts{i} = cat(3, grid_pts_i{:});
    end
end

function [color_Lab, color_XYZ, color_LMS] = convert_rgb_lab(param,...
    background_RGB, color_RGB)
    background_Spd = param.B_monitor*background_RGB;
    background_LMS = param.T_cones*background_Spd;
    background_XYZ = param.M_LMSToXYZ*background_LMS;

    %RGB -> SPD
    color_Spd      = param.B_monitor*color_RGB;
    %SPD -> LMS
    color_LMS      = param.T_cones*color_Spd;
    %LMS -> XYZ
    color_XYZ      = param.M_LMSToXYZ*color_LMS;
    %XYZ -> Lab
    color_Lab      = XYZToLab(color_XYZ, background_XYZ);
end

function deltaE = compute_deltaE(vecLen, background_RGB, ...
    ref_RGB, ref_Lab, vecDir, param)
    %compute comparison RGB
    comp_RGB = ref_RGB + vecDir.*vecLen;

    %convert it to Lab
    [comp_Lab, ~, ~] = convert_rgb_lab(param, background_RGB, comp_RGB);
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           PLOTTING FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_3D_RGBplanes(para, stm, plt)
    idx = plt.idx_fixed_RGB_slc;
    len_frames = length(idx);
    figure
    for l = 1:len_frames
        for p = 1:para.nPlanes
            subplot(1,para.nPlanes,p)
            %floor of the cube
            surf(para.x_grid, para.y_grid, zeros(para.nGridPts, para.nGridPts),...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.01); hold on
            %ceiling of the cube 
            surf(para.x_grid, para.y_grid, ones(para.nGridPts, para.nGridPts),...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.01);
            %walls of the cube
            surf(para.x_grid, ones(para.nGridPts, para.nGridPts), para.y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(para.x_grid, zeros(para.nGridPts, para.nGridPts), para.y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(zeros(para.nGridPts, para.nGridPts), para.x_grid, para.y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(ones(para.nGridPts, para.nGridPts), para.x_grid, para.y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            %wall edges
            plot3(zeros(1,para.nGridPts),zeros(1,para.nGridPts), para.grid,'k-');
            plot3(zeros(1,para.nGridPts),ones(1,para.nGridPts), para.grid,'k-');
            plot3(ones(1,para.nGridPts),zeros(1,para.nGridPts), para.grid,'k-');
            plot3(ones(1,para.nGridPts),ones(1,para.nGridPts), para.grid,'k-');
        
            %p = 1: GB plane; p = 2: RB plane; p = 3: RG plane
            surf(para.plane_points{idx(l)}{p}(:,:,1),para.plane_points{idx(l)}{p}(:,:,2),...
                para.plane_points{idx(l)}{p}(:,:,3), plt.colormapMatrix{idx(l)}{p},...
                'EdgeColor','none');
            %reference points
            scatter3(stm.ref_points{idx(l)}{p}(:,:,1),stm.ref_points{idx(l)}{p}(:,:,2),...
                stm.ref_points{idx(l)}{p}(:,:,3), 20,'k','Marker','+'); hold off
            xlim([0,1]); ylim([0,1]); zlim([0,1]); axis equal
            xlabel('R'); ylabel('G'); zlabel('B')
            xticks(0:0.2:1); yticks(0:0.2:1); zticks(0:0.2:1);
            title(plt.ttl{p});
        end
        set(gcf,'Units','normalized','Position',[0,0.1,0.7,0.4])
        set(gcf,'PaperUnits','centimeters','PaperSize',[30 12]);
        if plt.flag_save && len_frames > 1
            if l == 1; gif('RGB_cube.gif')
            else; gif
            end
        end
        pause(0.5)
    end
    if plt.flag_save && len_frames == 1 %1 frame
        set(gcf,'PaperUnits','centimeters','PaperSize',[30 12]);
        saveas(gcf, 'RGB_cube.pdf');
    end
end

function plot_isothreshold_contour(param, stim, results, plt)
    idx = plt.idx_fixed_RGB_slc;
    len_frames = length(idx);
    figure
    for l = 1:len_frames
        colormapMatrix2 = {[stim.fixed_RGBvec(idx(l)), 0, 1; 
                            stim.fixed_RGBvec(idx(l)), 0,0;...
                            stim.fixed_RGBvec(idx(l)), 1,0; 
                            stim.fixed_RGBvec(idx(l)), 1,1],...
                           [0, stim.fixed_RGBvec(idx(l)), 1; 
                            0,stim.fixed_RGBvec(idx(l)), 0;...
                            1,stim.fixed_RGBvec(idx(l)), 0; 
                            1,stim.fixed_RGBvec(idx(l)), 1],...
                           [0, 1, stim.fixed_RGBvec(idx(l)); 
                            0,0, stim.fixed_RGBvec(idx(l));...
                            1,0, stim.fixed_RGBvec(idx(l)); 
                            1,1, stim.fixed_RGBvec(idx(l))]};
        x = [0; 0; 1; 1];
        y = [1; 0; 0; 1];
        
        for p = 1:param.nPlanes
            subplot(1, param.nPlanes, p)
            patch('Vertices', [x,y],'Faces', [1 2 3 4], 'FaceVertexCData', ...
                colormapMatrix2{p}, 'FaceColor', 'interp'); hold on
            scatter(stim.x_grid_ref(:), stim.y_grid_ref(:), 20,'white','Marker','+');
        
            for i = 1:stim.nGridPts_ref
                for j = 1:stim.nGridPts_ref
                    %visualize the individual thresholds 
                    if plt.flag_visualizeRawData
                        scatter(squeeze(results.rgb_contour(idx(l),p,i,j,:,1)),...
                            squeeze(results.rgb_contour(idx(l),p,i,j,:,2)),10,...
                            'o','filled','MarkerEdgeColor',0.5.*ones(1,3),...
                            'MarkerFaceColor',0.5.*ones(1,3));
                    end
                    %visualize the best-fitting ellipse
                    plot(squeeze(results.fitEllipse(idx(l),p,i,j,:,1)),...
                        squeeze(results.fitEllipse(idx(l),p,i,j,:,2)),...
                        'white-','lineWidth',1.5)
                end
            end
            xlim([0,1]); ylim([0,1]); axis square; hold off
            xticks(0:0.2:1); yticks(0:0.2:1);
            title(plt.ttl{p})
            xlabel(plt.ttl{p}(1)); ylabel(plt.ttl{p}(2));
        end
        sgtitle(['The fixed other plane = ',num2str(stim.fixed_RGBvec(idx(l)))]);
        set(gcf,'Units','normalized','Position',[0,0.1,0.55,0.4]);
        set(gcf,'PaperUnits','centimeters','PaperSize',[30 12]);
        if plt.flag_save && len_frames > 1
            if l == 1; gif('Isothreshold_contour.gif')
            else; gif
            end
        end
        pause(1)
    end
    if plt.flag_save && len_frames == 1 %1 frame
        set(gcf,'PaperUnits','centimeters','PaperSize',[30 12]);
        saveas(gcf, 'Isothreshold_contour.pdf');
    end
end



