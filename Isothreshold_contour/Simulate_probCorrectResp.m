clear all; close all; clc

%% load isothreshold contours simulated based on CIELAB
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myDataDir   = 'Simulation_DataFiles';
intendedDir = fullfile(analysisDir, myDataDir);
addpath(intendedDir);

load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% First create a cube and select the RG, the RB and the GB planes
sim.slc_fixedVal     = 0.5; %the level of the fixed plane
sim.slc_fixedVal_idx = find(stim.fixed_RGBvec == sim.slc_fixedVal);

%get which plane to fix
slc_RGBplane         = input('Which plane would you like to fix (R/G/B):','s');
switch slc_RGBplane
    %GB plane with a fixed R value
    case 'R'; sim.slc_RGBplane = 1; 
    %RB plane with a fixed G value
    case 'G'; sim.slc_RGBplane = 2;
    %RG plane with a fixed B value
    case 'B'; sim.slc_RGBplane = 3;
end
sim.varying_RGBplane = setdiff(1:3, sim.slc_RGBplane);
sim.plane_points     = param.plane_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.ref_points       = stim.ref_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.background_RGB   = stim.background_RGB(:,sim.slc_fixedVal_idx);

%% Simulate data given the iso-threshold contours
sim.alpha                = 1.1729;
sim.beta                 = 1.2286;
sim.pC_given_alpha_beta  = ComputeWeibTAFC(stim.deltaE_1JND,sim.alpha,sim.beta);%0.8;
sim.nSims                = 360; %80; 240; 360
sim.random_jitter        = 0.1; %small jitter: 0.1; medium jitter: 0.2
sim.range_randomSampling = [-0.025, 0.025];
sim.method_sampling      = 'NearContour';

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_pij = squeeze(sim.ref_points(i,j,:));
        %convert it to Lab
        [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param.B_monitor, ...
            sim.background_RGB, param.T_cones, param.M_LMSToXYZ, rgb_ref_pij);
        sim.ref_Lab(i,j,:) = ref_Lab_lpij;
        
        %simulate comparison stimulus
        if strcmp(sim.method_sampling, 'NearContour')
            opt_vecLen_ij = squeeze(results.opt_vecLen(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane, i, j, :, :));
            ellPara = squeeze(results.ellParams(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane, i, j, :, :,:));
            sim.rgb_comp(i,j,:,:) = sample_rgb_comp_along2DChromDir(...
                rgb_ref_pij(sim.varying_RGBplane),...
                sim.varying_RGBplane, sim.slc_fixedVal, sim.nSims, ellPara,...
                sim.random_jitter);

            %fit an ellipse to simulated data
            [~, sim.fitEllipse_sampled_comp_unscaled(i,j,:,:), ~, ...
                sim.rgb_sample_comp_cov(i,j,:,:), sim.ellParams(i,j,:),...
                sim.AConstraint(i,j,:,:), sim.Ainv(i,j,:,:), sim.Q(i,j,:,:)] = ...
                fit_2d_isothreshold_contour(rgb_ref_pij, ...
                squeeze(sim.rgb_comp(i,j,:,:)), stim.grid_theta_xy,...
                'varyingRGBplane',sim.varying_RGBplane);

        elseif strcmp(sim.method_sampling, 'Random')
            sim.rgb_comp(i,j,:,:) = sample_rgb_comp_2Drandom(...
                rgb_ref_pij(sim.varying_RGBplane), sim.varying_RGBplane,...
                sim.slc_fixedVal, sim.range_randomSampling, sim.nSims);
        end
        
        %simulate binary responses
        for n = 1:sim.nSims
            [sim.lab_comp(i,j,:,n), ~, ~] = convert_rgb_lab(param.B_monitor, ...
                sim.background_RGB, param.T_cones, param.M_LMSToXYZ, ...
                squeeze(sim.rgb_comp(i,j,:,n)));
            sim.deltaE(i,j,n) = norm(squeeze(sim.lab_comp(i,j,:,n)) - ref_Lab_lpij);
            sim.probC(i,j,n) = ComputeWeibTAFC(sim.deltaE(i,j,n), ...
                sim.alpha, sim.beta);
        end
        sim.resp_binary(i,j,:) = binornd(1, squeeze(sim.probC(i,j,:)), [sim.nSims, 1]);
    end
end

%% visualize samples
flag_saveFig = false;
plt.ttl = {'GB plane', 'RB plane', 'RG plane'};
groundTruth_slc = squeeze(results.fitEllipse_unscaled(sim.slc_fixedVal_idx,...
    sim.slc_RGBplane,:,:,:,:));
figName = ['Sims_Isothreshold_contour_', plt.ttl{sim.slc_RGBplane},...
    '_sim',num2str(sim.nSims), 'perCond_sampling', sim.method_sampling]; 
if strcmp(sim.method_sampling, 'NearContour')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...%'groundTruth', groundTruth_slc,...
        'saveFig',flag_saveFig,...
        'figName',[figName,'_jitter',num2str(sim.random_jitter)]);
elseif strcmp(sim.method_sampling, 'Random')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...%'groundTruth', groundTruth_slc,...
        'responses',sim.resp_binary,...
        'saveFig',flag_saveFig, ...
        'figName',[figName,'_range',num2str(sim.range_randomSampling(end))]);    
end

%% save the data
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'Simulation_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end

if strcmp(sim.method_sampling, 'NearContour')
    fileName = ['Sims_isothreshold_',strtrim(plt.ttl{sim.slc_RGBplane}),'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),'.mat'];
elseif strcmp(sim.method_sampling, 'Random')
    fileName = ['Sims_isothreshold_',strtrim(plt.ttl{sim.slc_RGBplane}),'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),'.mat'];
end
outputName = fullfile(outputDir, fileName);
save(outputName,'sim');

