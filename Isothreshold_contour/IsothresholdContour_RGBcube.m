clear all; close all; clc

%% load stuff from psychtoolbox
% We'll spline all spectral functions to this common
% wavlength axis after loading them in.
S = [400 5 61];

% Load in LMS cone fundamentals
load T_cones_ss2.mat
param.T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);

% Load in primaries for a monitor
load B_monitor.mat
param.B_monitor = SplineSpd(S_monitor,B_monitor,S);
% M_RGBToLMS = T_cones*B_monitor;

% Load in XYZ CMFs 
load T_xyzCIEPhys2.mat
T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
param.M_LMSToXYZ = ((param.T_cones)'\(T_xyz)')';

%% First create a cube and select RG, RB and GB planes
param.nGridPts  = 100;
param.grid        = linspace(0, 1, param.nGridPts);
[param.x_grid, param.y_grid] = meshgrid(param.grid, param.grid);
param.nPlanes        = 3;

%1st plane: GB; 2nd plane: RB; 3rd plane: RG
param.plane_points = get_gridPts(param.x_grid, param.y_grid,...
    1:param.nPlanes, 0.5.*ones(1, param.nPlanes));

%set a grid for reference stimulus
stim.nGridPts_ref = 5;
stim.grid_ref     = linspace(0.2, 0.8, stim.nGridPts_ref);
[stim.x_grid_ref, stim.y_grid_ref] = meshgrid(stim.grid_ref,stim.grid_ref);
stim.ref_points = get_gridPts(stim.x_grid_ref, stim.y_grid_ref,...
    1:param.nPlanes, 0.5.*ones(1, param.nPlanes));

% Concatenate the R, G, B matrices
plt.ttl = {'GB plane', 'RB plane', 'RG plane'};
plt.colormapMatrix = param.plane_points;

%visualize the color planes
plot_3D_RGBplanes(param, stim, plt)

%% compute iso-threshold contour
%sample directions evenly
stim.background_RGB = ones(param.nPlanes,1).*0.5;
stim.numDirPts      = 17;
stim.grid_theta     = linspace(0, 2*pi,stim.numDirPts);
stim.grid_theta_xy  = [cos(stim.grid_theta(1:end-1)); ...
                       sin(stim.grid_theta(1:end-1))];
stim.deltaE_1JND    = 1;

results.contour_scaler = 5;
nThetaEllipse  = 200;
circleIn2D     = UnitCircleGenerate(nThetaEllipse);

for p = 1:param.nPlanes
    disp(p)
    vecDir = NaN(param.nPlanes,1); vecDir(p) = 0; 
    varChromDir = setdiff(1:param.nPlanes,p);
    for i = 1:stim.nGridPts_ref
        disp(i)
        for j = 1:stim.nGridPts_ref
            %grab the reference stimulus's RGB
            rgb_ref_pij = squeeze(stim.ref_points{p}(i,j,:));
            [results.ref_Lab(p,i,j,:), ~, ~] = convert_rgb_lab(param,...
                stim.background_RGB, rgb_ref_pij);

            %for each chromatic direction
            for k = 1:stim.numDirPts-1
                vecDir(varChromDir) = stim.grid_theta_xy(:,k);

                %run fmincon to search for the magnitude of vector that
                %leads to a deltaE of 1
                results.opt_vecLen(p,i,j,k) = find_vecLen(rgb_ref_pij, ...
                    squeeze(results.ref_Lab(p,i,j,:)), vecDir, param,stim);
            end
            % compute the iso-threshold contour 
            results.rgb_contour(p,i,j,:,:) = rgb_ref_pij(varChromDir)' + ...
                repmat(squeeze(results.opt_vecLen(p,i,j,:).*...
                results.contour_scaler), [1,2]).*stim.grid_theta_xy';

            %fit an ellipse
            [~,~,~,results.fitQ(p,i,j,:,:)] = FitEllipseQ(...
                squeeze(results.rgb_contour(p,i,j,:,:))' - ...
                rgb_ref_pij(varChromDir),'lockAngleAt0',false);
            results.fitEllipse(p,i,j,:,:) = (PointsOnEllipseQ(...
                squeeze(results.fitQ(p,i,j,:,:)),circleIn2D) +...
                rgb_ref_pij(varChromDir))';
        end
    end
end

%% visualize the iso-threshold contour
plot_isothreshold_contour(param, stim, results, plt)


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
    %convert
    background_Spd = param.B_monitor*background_RGB;
    background_LMS = param.T_cones*background_Spd;
    background_XYZ = param.M_LMSToXYZ*background_LMS;

    color_Spd = param.B_monitor*color_RGB;
    color_LMS = param.T_cones*color_Spd;
    color_XYZ = param.M_LMSToXYZ*color_LMS;
    color_Lab = XYZToLab(color_XYZ, background_XYZ);
end

function deltaE = compute_deltaE(vecLen, ref_RGB, ref_Lab, vecDir, param, stim)
    %compute comparison RGB
    comp_RGB = ref_RGB + vecDir.*vecLen;

    %convert it to Lab
    [comp_Lab, ~, ~] = convert_rgb_lab(param, stim.background_RGB, comp_RGB);
    deltaE = norm(comp_Lab - ref_Lab);
end

function opt_vecLen = find_vecLen(ref_RGB, ref_Lab, vecDir, param, stim)
    deltaE = @(d) abs(compute_deltaE(d, ref_RGB, ref_Lab, vecDir, param, stim)...
        - stim.deltaE_1JND);
    %have different initial points to avoid fmincon from getting stuck at
    %some places
    lb = 0; ub = 0.1;
    N_runs  = 2;
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
    figure
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
        surf(para.plane_points{p}(:,:,1),para.plane_points{p}(:,:,2),...
            para.plane_points{p}(:,:,3), plt.colormapMatrix{p},...
            'EdgeColor','none');
        %reference points
        scatter3(stm.ref_points{p}(:,:,1),stm.ref_points{p}(:,:,2),...
            stm.ref_points{p}(:,:,3), 20,'k','Marker','+'); 
        xlim([0,1]); ylim([0,1]); zlim([0,1]); axis equal
        xlabel('R'); ylabel('G'); zlabel('B')
        xticks(0:0.2:1); yticks(0:0.2:1); zticks(0:0.2:1);
        title(plt.ttl{p});
    end
    set(gcf,'Units','normalized','Position',[0,0,0.7,0.3])
end

function plot_isothreshold_contour(param, stim, results, plt)
    colormapMatrix2 = {[0.5, 0, 1; 0.5, 0,0; 0.5, 1,0; 0.5, 1,1],...
                       [0, 0.5, 1; 0,0.5, 0; 1,0.5, 0; 1,0.5, 1],...
                       [0, 1, 0.5; 0,0, 0.5; 1,0, 0.5; 1,1, 0.5]};
    x = [0; 0; 1; 1];
    y = [1; 0; 0; 1];
    
    figure
    for p = 1:param.nPlanes
        subplot(1, param.nPlanes, p)
        patch('Vertices', [x,y],'Faces', [1 2 3 4], 'FaceVertexCData', ...
            colormapMatrix2{p}, 'FaceColor', 'interp'); hold on
        scatter(stim.x_grid_ref(:), stim.y_grid_ref(:), 20,'white','Marker','+');
    
        varChromDir = setdiff(1:param.nPlanes,p);
        for i = 1:stim.nGridPts_ref
            for j = 1:stim.nGridPts_ref
                %visualize the individual thresholds 
                % scatter(squeeze(results.rgb_contour(p,i,j,:,1)),...
                % squeeze(resutls.rgb_contour(p,i,j,:,2)),5,'ko','filled');
                %visualize the best-fitting ellipse
                plot(squeeze(results.fitEllipse(p,i,j,:,1)),...
                    squeeze(results.fitEllipse(p,i,j,:,2)),...
                    'white-','lineWidth',1.5)
            end
        end
        xlim([0,1]); ylim([0,1]); axis square; 
        xticks(0:0.2:1); yticks(0:0.2:1);
        title(plt.ttl{p})
        xlabel(plt.ttl{p}(1)); ylabel(plt.ttl{p}(2));
    end
    set(gcf,'Units','normalized','Position',[0,0,0.5,0.3]);
end


