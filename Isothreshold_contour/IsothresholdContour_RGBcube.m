clear all; close all; clc

%% First create a cube and select RG, RB and GB planes
numGridPts = 100;
grid = linspace(0, 1, numGridPts);
[x_grid, y_grid] = meshgrid(grid, grid);
plane_points = {{ones(numGridPts, numGridPts).*0.5,x_grid, y_grid},...%GB plane
    {x_grid, ones(numGridPts, numGridPts).*0.5, y_grid},...%RB plane
    {x_grid, y_grid,ones(numGridPts, numGridPts).*0.5}};%RG plane

%set a grid for reference stimulus
numGridPts_ref = 5;
grid_ref = linspace(0.2, 0.8, numGridPts_ref);
[x_grid_ref, y_grid_ref] = meshgrid(grid_ref, grid_ref);
ref_points = {{ones(numGridPts_ref, numGridPts_ref).*0.5,x_grid_ref, y_grid_ref},... %GB plane
    {x_grid_ref, ones(numGridPts_ref, numGridPts_ref).*0.5,y_grid_ref},... %RB plane
    {x_grid_ref, y_grid_ref, ones(numGridPts_ref, numGridPts_ref).*0.5}}; %RG plane

% Concatenate the R, G, B matrices
ttl = {'GB plane', 'RB plane', 'RG plane'};
colormapMatrix = {cat(3, ones(numGridPts, numGridPts).*0.5, x_grid, y_grid),...
    cat(3, x_grid, ones(numGridPts, numGridPts).*0.5, y_grid),...
    cat(3, x_grid, y_grid, ones(numGridPts, numGridPts).*0.5)};

%% plotting
figure
for p = 1:length(ttl)
    subplot(1,length(ttl),p)
    %floor of the cube
    surf(x_grid, y_grid, zeros(numGridPts, numGridPts),'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.01); hold on
    %ceiling of the cube 
    surf(x_grid, y_grid, ones(numGridPts, numGridPts),'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.01);
    %walls of the cube
    surf(x_grid, ones(numGridPts, numGridPts), y_grid,'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.05);
    surf(x_grid, zeros(numGridPts, numGridPts), y_grid,'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.05);
    surf(zeros(numGridPts, numGridPts), x_grid, y_grid,'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.05);
    surf(ones(numGridPts, numGridPts), x_grid, y_grid,'FaceColor','k', ...
        'EdgeColor','none','FaceAlpha',0.05);
    %wall edges
    plot3(zeros(1,numGridPts),zeros(1,numGridPts), grid,'k-');
    plot3(zeros(1,numGridPts),ones(1,numGridPts), grid,'k-');
    plot3(ones(1,numGridPts),zeros(1,numGridPts), grid,'k-');
    plot3(ones(1,numGridPts),ones(1,numGridPts), grid,'k-');

    %p = 1: GB plane; p = 2: RB plane; p = 3: RG plane
    surf(plane_points{p}{1}, plane_points{p}{2},plane_points{p}{3},...
        colormapMatrix{p},'EdgeColor','none');
    %reference points
    scatter3(ref_points{p}{1},ref_points{p}{2}, ref_points{p}{3},...
        20,'k','Marker','+'); 
    xlim([0,1]); ylim([0,1]); zlim([0,1]); axis equal
    xlabel('R'); ylabel('G'); zlabel('B')
    title(ttl{p});
end
set(gcf,'Units','normalized','Position',[0,0,0.7,0.3])

%% compute iso-threshold contour
%sample directions evenly
global background_RGB deltaE_1JND
background_RGB = ones(3,1).*0.5;
numDirPts      = 17;
theta_grid     = linspace(0, 2*pi,numDirPts);
theta_xy_grid  = [cos(theta_grid(1:end-1)); sin(theta_grid(1:end-1))];
delta_inc      = linspace(1e-4, 1e-3, 20);
deltaE_1JND     = 1;

ref_Lab        = NaN(length(ttl), numGridPts_ref, numGridPts_ref, 3);
opt_vecLen     = NaN(length(ttl), numGridPts_ref, numGridPts_ref, numDirPts-1);
for p = 1:length(ttl)
    disp(p)
    varChromDir = setdiff(1:length(ttl),p);
    for i = 1:numGridPts_ref
        disp(i)
        for j = 1:numGridPts_ref
            %grab the reference stimulus's RGB
            rgb_ref_pij = [ref_points{p}{1}(i,j); ref_points{p}{2}(i,j);...
                ref_points{p}{3}(i,j)];
            [ref_Lab(p,i,j,:), ~, ~] = convert_rgb_lab(background_RGB, rgb_ref_pij);
            for k = 1:numDirPts-1
                theta_xy_grid_k = theta_xy_grid(:,k);
                vecDir = NaN(3,1); vecDir(p) = 0; vecDir(varChromDir) = theta_xy_grid_k;

                %run fmincon to search for the magnitude of vector that
                %leads to a deltaE of 1
                opt_vecLen(p,i,j,k) = find_vecLen(rgb_ref_pij, squeeze(ref_Lab(p,i,j,:)), vecDir);
            end
        end
    end
end

%% compute the iso-threshold contour and fit an ellipse
contour_scaler = 5;
nThetaEllipse = 200;
circleIn2D = UnitCircleGenerate(nThetaEllipse);
fitQ = NaN(length(ttl), numGridPts_ref, numGridPts_ref, 2, 2);
rgb_contour = NaN(length(ttl), numGridPts_ref, numGridPts_ref, numDirPts-1, 2);
fitEllipse = NaN(length(ttl), numGridPts_ref, numGridPts_ref, 2, nThetaEllipse);
for p = 1:length(ttl)
    varChromDir = setdiff(1:length(ttl),p);
    for i = 1:numGridPts_ref
        for j = 1:numGridPts_ref
            rgb_ref_pij = [ref_points{p}{1}(i,j); ref_points{p}{2}(i,j);...
                ref_points{p}{3}(i,j)];
            rgb_contour(p,i,j,:,:) = rgb_ref_pij(varChromDir)' + ...
                repmat(squeeze(opt_vecLen(p,i,j,:).*contour_scaler),[1,2]).*theta_xy_grid';
            %fit an ellipse
            [~,~,~,fitQ(p,i,j,:,:)] = FitEllipseQ(...
                squeeze(rgb_contour(p,i,j,:,:))' - rgb_ref_pij(varChromDir),'lockAngleAt0',false);
            fitEllipse(p,i,j,:,:) = PointsOnEllipseQ(squeeze(fitQ(p,i,j,:,:)),circleIn2D) +...
                rgb_ref_pij(varChromDir);
        end
    end
end

%% visualize the iso-threshold contour
colormapMatrix2 = {[0.5, 0, 1; 0.5, 0,0; 0.5, 1,0; 0.5, 1,1],...
    [0,0.5,  1;0,0.5, 0; 1,0.5, 0; 1,0.5, 1],...
    [0, 1,0.5;0,0,0.5; 1,0,0.5; 1,1,0.5]};
x = [0; 0; 1; 1];
y = [1; 0; 0; 1];

figure
for p = 1:length(ttl)
    subplot(1, length(ttl), p)
    patch('Vertices', [x,y],'Faces', [1 2 3 4], 'FaceVertexCData', ...
        colormapMatrix2{p}, 'FaceColor', 'interp'); hold on
    scatter(x_grid_ref(:), y_grid_ref(:), 20,'white','Marker','+');

    varChromDir = setdiff(1:length(ttl),p);
    for i = 1:numGridPts_ref
        for j = 1:numGridPts_ref
            %visualize the individual thresholds 
            % scatter(squeeze(rgb_contour(p,i,j,:,1)), squeeze(rgb_contour(p,i,j,:,2)),5,'ko','filled');
            %visualize the best-fitting ellipse
            plot(squeeze(fitEllipse(p,i,j,1,:)), squeeze(fitEllipse(p,i,j,2,:)),'white-')
        end
    end

    xlim([0,1]); ylim([0,1]); axis square;
    title(ttl{p})
end



%% HELPING FUNCTIONS
function [color_Lab, color_XYZ, color_LMS] = convert_rgb_lab(background_RGB, color_RGB)
    % We'll spline all spectral functions to this common
    % wavlength axis after loading them in.
    S = [400 5 61];

    % Data in Psychtoolbox
    load T_cones_ss2.mat
    T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);

    load B_monitor.mat
    B_monitor = SplineSpd(S_monitor,B_monitor,S);
    M_RGBToLMS = T_cones*B_monitor;

    % Data in Psychtoolbox
    load T_xyzCIEPhys2.mat
    T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
    M_LMSToXYZ = ((T_cones)'\(T_xyz)')';

    %convert
    background_Spd = B_monitor*background_RGB;
    background_LMS = T_cones*background_Spd;
    background_XYZ = M_LMSToXYZ*background_LMS;

    color_Spd = B_monitor*color_RGB;
    color_LMS = T_cones*color_Spd;
    color_XYZ = M_LMSToXYZ*color_LMS;
    color_Lab = XYZToLab(color_XYZ, background_XYZ);
end


function deltaE = compute_deltaE(vecLen, ref_RGB, ref_Lab, vecDir)
    global background_RGB
    %compute comparison RGB
    comp_RGB = ref_RGB + vecDir.*vecLen;

    %convert it to Lab
    [comp_Lab, ~, ~] = convert_rgb_lab(background_RGB, comp_RGB);
    deltaE = norm(comp_Lab - ref_Lab);
end

function opt_vecLen = find_vecLen(ref_RGB, ref_Lab, vecDir)
    global deltaE_1JND
    deltaE = @(d) abs(compute_deltaE(d, ref_RGB, ref_Lab, vecDir) - deltaE_1JND);
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



