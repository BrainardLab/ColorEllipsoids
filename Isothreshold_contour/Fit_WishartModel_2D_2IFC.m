clear all; close all; clc

%% load isothreshold contours simulated based on CIELAB
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myDataDir   = 'Simulation_DataFiles';
intendedDir = fullfile(analysisDir, myDataDir);
addpath(intendedDir);
load('Sims_isothreshold_RG plane_sim240perCond_samplingNearContour_jitter0.1.mat', 'sim')
% load('Sims_isothreshold_GB plane_sim2000perCond_samplingRandom_range0.025.mat', 'sim')
load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% define model specificity for the wishart process
model.max_deg           = 3;     %corresponds to p in Alex's document
model.nDims             = 2;     %corresponds to a = 1, a = 2 
model.eDims             = 0;     %extra dimensions
model.num_grid_pts      = 100;
model.num_MC_samples    = 100;   %number of monte carlo simulation

model.coeffs_chebyshev = compute_chebyshev_basis_coeffs(model.max_deg);
disp(model.coeffs_chebyshev)

model.xt = linspace(-1,1,model.num_MC_samples);
model.yt = linspace(-1,1,model.num_MC_samples);
[model.XT, model.YT]   = meshgrid(model.xt, model.yt);
XYT = cat(3, model.XT, model.YT);
[~, model.M_chebyshev] = compute_U(model.coeffs_chebyshev,[],...
    XYT, model.max_deg);
%visualize it
plot_multiHeatmap(model.M_chebyshev,'permute_M',true);

%% Fit the model to the data
%comparison stimulus
%x_sim: stim.nGridPts_ref x stim.nGridPts_ref x dims x sim.nSims

sim.slc_ref                 = 1:stim.nGridPts_ref;
%select only a subset of reference stimulus
% sim.slc_ref             = 2:1:4;%1:2:stim.nGridPts_ref;
x_sim                   = sim.rgb_comp(sim.slc_ref,sim.slc_ref,sim.varying_RGBplane,:);
x_sim_d1                = squeeze(x_sim(:,:,1,:));
x_sim_d2                = squeeze(x_sim(:,:,2,:));
sim.x_sim_org(1,:,1)    = x_sim_d1(:); 
sim.x_sim_org(1,:,2)    = x_sim_d2(:);

%reference stimulus
xbar_sim                = repmat(sim.ref_points(sim.slc_ref,sim.slc_ref,sim.varying_RGBplane),...
                             [1,1,1,sim.nSims]);
xbar_sim_d1             = squeeze(xbar_sim(:,:,1,:));
xbar_sim_d2             = squeeze(xbar_sim(:,:,2,:));
sim.xbar_sim_org(1,:,1) = xbar_sim_d1(:); 
sim.xbar_sim_org(1,:,2) = xbar_sim_d2(:);
% if we just want to use one set of randomly drawn samples
% sim.etas = 0.01.*randn([2,1,model.nDims + model.eDims, model.num_MC_samples]);

%response
resp_binary = sim.resp_binary(sim.slc_ref, sim.slc_ref,:);

%define objective functions
w_reshape_size = [model.max_deg, model.max_deg, model.nDims, model.nDims+model.eDims];
objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec,w_reshape_size,...
    sim.x_sim_org, sim.xbar_sim_org, resp_binary(:), ...
    model.coeffs_chebyshev,'num_MC_samples', model.num_MC_samples);

%% call bads 
fits.nFreeParams = model.max_deg*model.max_deg*model.nDims*...
                    (model.nDims + model.eDims);
fits.lb          = -0.05.*ones(1, fits.nFreeParams); %-0.05
fits.ub          =  0.05.*ones(1, fits.nFreeParams); %0.05
fits.plb         = -0.01.*ones(1, fits.nFreeParams); %-0.01
fits.pub         =  0.01.*ones(1, fits.nFreeParams); %0.01
%have different initial points to avoid fmincon from getting stuck at
%some places
fits.N_runs      = 3;
fits.init        = rand(fits.N_runs,fits.nFreeParams).*...
                    (fits.pub-fits.plb) + fits.plb;

for n = 1:fits.N_runs
    disp(n)
    %use fmincon to search for the optimal defocus
    [fits.w_colvec_est(n,:), fits.minVal(n)] = bads(objectiveFunc,...
        fits.init(n,:), fits.lb, fits.ub, fits.plb, fits.pub);
end
%find the index that corresponds to the minimum value
[~,idx_min] = min(fits.minVal);
%find the corresponding optimal focus that leads to the highest peak of
%the psf's
fits.w_colvec_est_best  = fits.w_colvec_est(idx_min,:);
fits.w_est_best = reshape(fits.w_colvec_est_best, [model.max_deg,...
    model.max_deg, model.nDims, model.nDims+model.eDims]);

%% recover covariance matrix
xy_grid = cat(3,param.x_grid, param.y_grid);
[fits.U_recover,~] = compute_U(model.coeffs_chebyshev, fits.w_est_best, ...
    xy_grid, model.max_deg); 
for i = 1:model.nDims
    for j = 1:model.nDims
        fits.Sigmas_recover(:,:,i,j) = sum(fits.U_recover(:,:,i,:).*...
            fits.U_recover(:,:,j,:),4);
    end
end

%% make predictions on the probability of correct
fits.ngrid_bruteforce = 2e3;
fits.vecLength = linspace(0,max(results.opt_vecLen(:))*1.2,fits.ngrid_bruteforce);

%for each reference stimulus
for i = 1:length(sim.slc_ref) %stim.nGridPts_ref
    for j = 1:length(sim.slc_ref) %stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_ij = sim.ref_points(sim.slc_ref(i),sim.slc_ref(j),:);

        [fits.recover_fitEllipse(i,j,:,:), ...
            fits.recover_fitEllipse_unscaled(i,j,:,:), ...
            fits.recover_rgb_contour(i,j,:,:), ...
            fits.recover_rgb_contour_cov(i,j,:,:), ...
            fits.recover_rgb_comp_est(i,j,:,:)] = ...
            convert_Sig_2DisothresholdContour(rgb_ref_ij, sim.varying_RGBplane, ...
            stim.grid_theta_xy, fits.vecLength, sim.pC_given_alpha_beta, ...
            model.coeffs_chebyshev, fits.w_est_best, 'contour_scaler',...
            results.contour_scaler, 'nSteps_bruteforce', fits.ngrid_bruteforce);
    end
end

%reshape the matrices for plotting
recover_fitEllipse_plt = NaN([1,1,size(fits.recover_fitEllipse)]);
recover_fitEllipse_plt(1,1,:,:,:,:) = fits.recover_fitEllipse;

%% extrapolate
fits.grid_ref_extrap = stim.grid_ref(1:end-1) + diff(stim.grid_ref(1:2))/2;
fits.nGridPts_ref = length(fits.grid_ref_extrap);
[fits.x_grid_ref_extrap, fits.y_grid_ref_extrap] = meshgrid(fits.grid_ref_extrap, fits.grid_ref_extrap);
fits.ref_points_extrap = get_gridPts(fits.x_grid_ref_extrap,...
    fits.y_grid_ref_extrap, sim.slc_RGBplane, sim.slc_fixedVal);

% fits.grid_ref_extrap = stim.grid_ref;
% fits.nGridPts_ref = length(fits.grid_ref_extrap);
% [fits.x_grid_ref_extrap, fits.y_grid_ref_extrap] = meshgrid(fits.grid_ref_extrap, fits.grid_ref_extrap);
% fits.ref_points_extrap = get_gridPts(fits.x_grid_ref_extrap,...
%     fits.y_grid_ref_extrap, sim.slc_RGBplane, sim.slc_fixedVal);

%for each reference stimulus
for i = 1:fits.nGridPts_ref
    for j = 1:fits.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_ij = fits.ref_points_extrap{1}(i,j,:);

        [fits.extrap_fitEllipse(i,j,:,:), ...
            fits.extrap_fitEllipse_unscaled(i,j,:,:), ...
            fits.extrap_rgb_contour(i,j,:,:), ...
            fits.extrap_rgb_contour_cov(i,j,:,:), ...
            fits.extrap_rgb_comp_est(i,j,:,:),...
            fits.extrap_ellParams(i,j,:),...
            fits.extrap_AConstraint(i,j,:,:),...
            fits.extrap_Ainv(i,j,:,:),...
            fits.extrap_Q(i,j,:,:)] = ...
            convert_Sig_2DisothresholdContour(rgb_ref_ij, sim.varying_RGBplane, ...
            stim.grid_theta_xy, fits.vecLength, sim.pC_given_alpha_beta, ...
            model.coeffs_chebyshev, fits.w_est_best, 'contour_scaler',...
            results.contour_scaler, 'nSteps_bruteforce', fits.ngrid_bruteforce);
    end
end

extrap_fitEllipse_plt = NaN([1,1,size(fits.extrap_fitEllipse)]);
extrap_fitEllipse_plt(1,1,:,:,:,:) = fits.extrap_fitEllipse;

%% visualize
plt.ttl = {'GB plane', 'RB plane', 'RG plane'};
if length(sim.slc_ref) < stim.nGridPts_ref
    str_extension = ['_subsetRef',num2str(sim.slc_ref)];
else; str_extension = ''; 
end
if strcmp(sim.method_sampling,'NearContour')
    figName = ['Fitted_Isothreshold_contour_',plt.ttl{sim.slc_RGBplane},...
        '_sim',num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),str_extension];
elseif strcmp(sim.method_sampling, 'Random')
    figName = ['Fitted_Isothreshold_contour_',plt.ttl{sim.slc_RGBplane},...
        '_sim',num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),str_extension];
end

plot_2D_isothreshold_contour(stim.x_grid_ref, stim.y_grid_ref, ...
    results.fitEllipse(sim.slc_fixedVal_idx,...
    sim.slc_RGBplane,:,:,:,:),sim.slc_fixedVal,...
    'slc_x_grid_ref',sim.slc_ref,...
    'slc_y_grid_ref',sim.slc_ref,...
    'WishartEllipses',recover_fitEllipse_plt, ... 
    'ExtrapEllipses',[],...%extrap_fitEllipse_plt,...
    'rgb_background',false,...
    'xlabel',plt.ttl{sim.slc_RGBplane}(1),...
    'ylabel',plt.ttl{sim.slc_RGBplane}(2),...
    'refColor',[0,0,0],...
    'EllipsesColor',[178,34,34]./255,...
    'WishartEllipsesColor',[76,153,0]./255,...
    'ExtrapEllipsesColor',[0.5,0.5,0.5],...
    'figPos',[0,0.1,0.35,0.4],...
    'subTitle', {sprintf(['Predicted iso-threshold contours \nin the ',...
        plt.ttl{sim.slc_RGBplane}, ' based on the Wishart process'])},...
    'figName', figName,...
    'saveFig',false);

%% visualize samples
groundTruth_slc = squeeze(results.fitEllipse_unscaled(sim.slc_fixedVal_idx,...
    sim.slc_RGBplane,:,:,:,:));
% groundTruth_slc = sim.fitEllipse_sampled_comp_unscaled;

if strcmp(sim.method_sampling, 'NearContour')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...
        'slc_x_grid_ref',sim.slc_ref,...
        'slc_y_grid_ref',sim.slc_ref,...
        'EllipsesColor',[178,34,34]./255,...
        'WishartEllipsesColor',[76,153,0]./255,...
        'groundTruth', groundTruth_slc,...
        'modelPredictions',fits.recover_fitEllipse_unscaled,...
        'saveFig',true,...
        'figName',[figName, '_visualizeSamples',str_extension])
elseif strcmp(sim.method_sampling, 'Random')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...
        'EllipsesColor',[178,34,34]./255,...
        'WishartEllipsesColor',[76,153,0]./255,...
        'responses', sim.resp_binary,...
        'groundTruth',groundTruth_slc,...
        'modelPredictions', fits.recover_fitEllipse_unscaled, ...
        'saveFig',true,...
        'figName',[figName, '_visualizeSamples',str_extension])    
end

%% save the data
E = {param, stim, results, plt, sim, model, fits};
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'ModelFitting_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end

if strcmp(sim.method_sampling, 'NearContour')
    fileName = ['Fits_isothreshold_',plt.ttl{sim.slc_RGBplane},'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),str_extension,'.mat'];
elseif strcmp(sim.method_sampling, 'Random')
    fileName = ['Fits_isothreshold_',plt.ttl{sim.slc_RGBplane},'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),str_extension,'.mat'];
end
outputName = fullfile(outputDir, fileName);
save(outputName,'E');
