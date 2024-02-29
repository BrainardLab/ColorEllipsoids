clear all; close all; clc; 
seed = rng(1);

%% load isothreshold contours simulated based on CIELAB
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myDataDir   = 'Simulation_DataFiles';
intendedDir = fullfile(analysisDir, myDataDir);
addpath(intendedDir);

load('Isothreshold_ellipsoid_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% First create a cube and select the RG, the RB and the GB planes
sim.varying_RGBplane = 1:3;
sim.ref_points       = stim.ref_points;
sim.background_RGB   = stim.background_RGB;

%% Simulate data given the iso-threshold contours
sim.alpha                = 1.1729;
sim.beta                 = 1.2286;
sim.pC_given_alpha_beta  = ComputeWeibTAFC(stim.deltaE_1JND,sim.alpha,sim.beta);
sim.nSims                = 480; %40; 80; 240; 360
sim.random_jitter        = 0.1; %small jitter: 0.1; medium jitter: 0.2
sim.range_randomSampling = [-0.025, 0.025];
sim.method_sampling      = 'NearContour'; %'NearContour'

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        for k = 1:stim.nGridPts_ref
            %grab the reference stimulus's RGB
            rgb_ref_ijk = squeeze(sim.ref_points(i,j,k,:));
            %convert it to Lab
            [ref_Lab_ijk, ~, ~] = convert_rgb_lab(param.B_monitor, ...
                sim.background_RGB, param.T_cones, param.M_LMSToXYZ, rgb_ref_ijk);
            sim.ref_Lab(i,j,k,:) = ref_Lab_ijk;
            
            %simulate comparison stimulus
            if strcmp(sim.method_sampling, 'NearContour')
                ellPara = results.ellParams{i, j, k};
                sim.rgb_comp(i,j,k,:,:) = sample_rgb_comp_3DNearContour(...
                    rgb_ref_ijk, sim.nSims, ellPara{2}, ellPara{3}, sim.random_jitter);
    
                %fit an ellipse to simulated data
                [~, sim.fitEllipsoid_sampled_comp_unscaled(i,j,k,:,:), ~, ...
                    sim.rgb_surface_cov(i,j,k,:,:), sim.ellParams{i,j,k}] = ...
                    fit_3d_isothreshold_ellipsoid(rgb_ref_ijk, ...
                    squeeze(sim.rgb_comp(i,j,k,:,:))', stim.grid_xyz,...
                    'varyingRGBplane',1:3,...
                    'nThetaEllipse',plt.nThetaEllipse,...
                    'nPhiEllipse',plt.nPhiEllipse,...
                    'ellipse_scaler',results.contour_scaler);
            elseif strcmp(sim.method_sampling, 'Random')
                sim.rgb_comp(i,j,:,:) = sample_rgb_comp_2Drandom(...
                    rgb_ref_ijk(sim.varying_RGBplane), sim.varying_RGBplane,...
                    sim.slc_fixedVal, sim.range_randomSampling, sim.nSims);
            end
            
            %simulate binary responses
            for n = 1:sim.nSims
                [sim.lab_comp(i,j,k,:,n), ~, ~] = convert_rgb_lab(param.B_monitor, ...
                    sim.background_RGB, param.T_cones, param.M_LMSToXYZ, ...
                    squeeze(sim.rgb_comp(i,j,k,:,n)));
                sim.deltaE(i,j,k,n) = norm(squeeze(sim.lab_comp(i,j,k,:,n)) - ref_Lab_ijk);
                sim.probC(i,j,k,n) = ComputeWeibTAFC(sim.deltaE(i,j,k,n), ...
                    sim.alpha, sim.beta);
            end
            sim.resp_binary(i,j,k,:) = binornd(1, squeeze(sim.probC(i,j,k,:)), [sim.nSims, 1]);
        end
    end
end

%% visualize samples
idx_x = 3; idx_y = 5; idx_z = 1;
figure
tiledlayout(3,3,'TileSpacing','none');
for i = 1:2:stim.nGridPts_ref
    for j = 1:2:stim.nGridPts_ref
        nexttile
        %subplot(3,3,3*floor(i/2)+ceil(j/2))
        slc_ref = stim.ref_points(idx_x, i, j, :);
        slc_gt = squeeze(results.fitEllipsoid_unscaled(idx_x,i, j,:,:));
        slc_gt_x = reshape(slc_gt(:,1), [plt.nPhiEllipse, plt.nThetaEllipse]);
        slc_gt_y = reshape(slc_gt(:,2), [plt.nPhiEllipse, plt.nThetaEllipse]);
        slc_gt_z = reshape(slc_gt(:,3), [plt.nPhiEllipse, plt.nThetaEllipse]);
        slc_rgb_comp = squeeze(sim.rgb_comp(idx_x, i,j,:,:));
        
        scatter3(slc_rgb_comp(1,:), slc_rgb_comp(2,:), slc_rgb_comp(3,:),10,...
            'MarkerFaceColor', [0,0,0],...
            'MarkerEdgeColor','none','MarkerFaceAlpha',0.3);hold on
        surf(slc_gt_x,slc_gt_y,slc_gt_z,'FaceColor', ...
            [stim.grid_ref(idx_x), stim.grid_ref(i), stim.grid_ref(j)],...
            'EdgeColor','none','FaceAlpha',0.4)
        camlight right; lighting phong;
        xlim(slc_ref(1)+[-0.025,0.025]);
        ylim(slc_ref(2)+[-0.025,0.025]);
        zlim(slc_ref(3)+[-0.025,0.025]); 
        axis square
        xticks(slc_ref(1));
        yticks(slc_ref(2));
        zticks(slc_ref(3));
        % if i == 3 && j == 3; xlabel('R'); ylabel('G'); zlabel('B');end
        view(-90,0) %view from leftside
    end
end
set(gcf,'Units','Normalized','Position',[0,0.1,0.425,1])

%% save the data
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir    = 'Simulation_DataFiles/DataFiles_HPC';
outputDir   = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end

%whether we want to save the seed
flag_saveSeed = true;
if flag_saveSeed; str_extension = ['_rng', num2str(seed.Seed)];
else; str_extension = ''; end

if strcmp(sim.method_sampling, 'NearContour')
    fileName = ['Sims_isothreshold_',strtrim(plt.ttl{sim.slc_RGBplane}),'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),str_extension,'.mat'];
elseif strcmp(sim.method_sampling, 'Random')
    fileName = ['Sims_isothreshold_',strtrim(plt.ttl{sim.slc_RGBplane}),'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),str_extension,'.mat'];
end
outputName = fullfile(outputDir, fileName);
save(outputName,'sim');
