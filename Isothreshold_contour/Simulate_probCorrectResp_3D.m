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

%% Simulate data given the iso-threshold contours
sim.ref_points           = stim.ref_points;
sim.background_RGB       = stim.background_RGB;
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
                ellPara = results.ellipsoidParams{i, j, k};
                sim.rgb_comp(i,j,k,:,:) = sample_rgb_comp_3DNearContour(...
                    rgb_ref_ijk, sim.nSims, ellPara{2}, ellPara{3}, sim.random_jitter);
    
                %fit an Ellipsoid to simulated data
                [~, sim.fitEllipsoid_sampled_comp_unscaled(i,j,k,:,:), ~, ...
                    sim.rgb_surface_cov(i,j,k,:,:), sim.ellParams{i,j,k}] = ...
                    fit_3d_isothreshold_ellipsoid(rgb_ref_ijk, ...
                    squeeze(sim.rgb_comp(i,j,k,:,:))', stim.grid_xyz,...
                    'nThetaEllipsoid',plt.nThetaEllipsoid,...
                    'nPhiEllipsoid',plt.nPhiEllipsoid,...
                    'Ellipsoid_scaler',results.ellipsoid_scaler);
            elseif strcmp(sim.method_sampling, 'Random')
                sim.rgb_comp(i,j,:,:) = sample_rgb_comp_3Drandom(...
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

%%
slc_plane   = 'GB plane'; %RG plane; RB plane; GB plane
fixed_plane = 'R';        %B         G         R
visualize_ellipse = false;
if visualize_ellipse; str_ext = 'ellipses_'; else; str_ext = ''; end
figName = ['Sims_Isothreshold_ellipsoids_',str_ext, slc_plane,...
    '_sim',num2str(sim.nSims), 'perCond_sampling', sim.method_sampling]; 

plot_3D_sampledComp(stim.grid_ref, results.fitEllipsoid_unscaled,sim.rgb_comp,...
    fixed_plane, 0.5, plt.nPhiEllipsoid, plt.nThetaEllipsoid, results.fitEllipse_unscaled,... 
    'visualize_ellipse', visualize_ellipse,...
    'visualize_samples', true,...
    'slc_grid_ref_dim1',1:2:5,...
    'slc_grid_ref_dim2',1:2:5,...
    'samples_alpha',0.3,...
    'default_viewing_angle',false,...%'surf_alpha',0.2,...
    'lineWidth_ellipse',3,...
    'fontSize',12,...
    'saveFig',true, ...
    'title',slc_plane,...
    'paperSize',[45,45],...
    'figName', [figName,'_jitter',num2str(sim.random_jitter)])

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
    fileName = ['Sims_isothreshold_ellipsoids_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),str_extension,'.mat'];
elseif strcmp(sim.method_sampling, 'Random')
    fileName = ['Sims_isothreshold_ellipsoids_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),str_extension,'.mat'];
end
outputName = fullfile(outputDir, fileName);
save(outputName,'sim');
