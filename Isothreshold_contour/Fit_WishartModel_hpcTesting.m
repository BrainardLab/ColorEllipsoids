clear all; close all; clc; parpool(3);
%This script will be run in HPC
% addpath(genpath('/home/fh862/bads-master'));
% addpath(genpath('/home/fh862/DB_projects/WPModel_testing/DataFiles'));
% addpath(genpath('/home/fh862/DB_projects/WPModel_tensorproduct'));

%% define model specificity for the wishart process
model.max_deg           = 3;     %corresponds to p in Alex's document
model.nDims             = 2;     %corresponds to a = 1, a = 2 
model.eDims             = 0;     %extra dimensions
model.scaling_elevation = -1;
model.scaling_streching = 2;
model.num_grid_pts      = 100;
model.num_MC_samples    = 100;   %number of monte carlo simulation
%compute chebyshev polynomials
model.coeffs_chebyshev = compute_chebyshev_basis_coeffs(model.max_deg);
model.xt               = linspace(-1,1,model.num_MC_samples);
model.yt               = linspace(-1,1,model.num_MC_samples);
[model.XT, model.YT]   = meshgrid(model.xt, model.yt);
[~, model.M_chebyshev] = compute_U(model.coeffs_chebyshev,[],...
    model.XT,model.YT, model.max_deg);

%% load isothreshold contours simulated based on CIELAB
% analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
% myDataDir   = 'Simulation_DataFiles/DataFiles_HPC';
% intendedDir = fullfile(analysisDir, myDataDir);
% addpath(intendedDir);
load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%we have a total of nDataFiles, each of which was generated by a different
%seed
nDataFiles = 10;
for s = 1:nDataFiles
    load(['Sims_isothreshold_GB plane_sim40perCond_samplingNearContour_',...
        'jitter0.1_rng',num2str(s),'.mat'], 'sim');
    
    %% Fit the model to the data
    %comparison stimulus
    %x_sim: stim.nGridPts_ref x stim.nGridPts_ref x dims x sim.nSims
    sim.slc_ref             = 1:stim.nGridPts_ref;
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
    
    %response
    resp_binary = sim.resp_binary(sim.slc_ref, sim.slc_ref,:);
    
    %define objective functions
    w_reshape_size = [model.max_deg, model.max_deg, model.nDims, model.nDims+model.eDims];
    objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec,w_reshape_size,...
        sim.x_sim_org, sim.xbar_sim_org, resp_binary(:), ...
        model.coeffs_chebyshev,'num_MC_samples', model.num_MC_samples);
    
    % call bads 
    %compute the total number of free parameters
    fits.nFreeParams = model.max_deg*model.max_deg*model.nDims*(model.nDims + model.eDims);
    %lower and upper founds for searching the best-fitting parameters
    lb  = -0.05.*ones(1, fits.nFreeParams); 
    ub  =  0.05.*ones(1, fits.nFreeParams); 
    plb = -0.01.*ones(1, fits.nFreeParams); 
    pub =  0.01.*ones(1, fits.nFreeParams); 
    %have different initial points to avoid fmincon from getting stuck at
    %some places
    fits.N_runs = 3;
    init = rand(fits.N_runs,fits.nFreeParams).*(pub-plb) + plb;
    
    %initialize 
    w_colvec_est = NaN(fits.N_runs, fits.nFreeParams);
    minVal = NaN(1, fits.N_runs);
    parfor n = 1:fits.N_runs
        disp(n)
        %use bads to search for the optimal defocus
        [w_colvec_est(n,:), minVal(n)] = bads(objectiveFunc, init(n,:), lb, ub, plb, pub);
    end
    %save bounds and the best-fitting parameters
    fits.w_colvec_est = w_colvec_est;
    fits.minVal = minVal;
    fits.init   = init;
    fits.lb     = lb;
    fits.ub     = ub;
    fits.plb    = plb;
    fits.pub    = pub;
    %find the index that corresponds to the minimum value
    [~,idx_min] = min(fits.minVal);
    %find the corresponding optimal focus that leads to the highest peak of
    %the psf's
    fits.w_colvec_est_best  = fits.w_colvec_est(idx_min,:);
    fits.w_est_best = reshape(fits.w_colvec_est_best, [model.max_deg,...
        model.max_deg, model.nDims, model.nDims+model.eDims]);
    
    %% recover covariance matrix
    [fits.U_recover,~] = compute_U(model.coeffs_chebyshev, fits.w_est_best, ...
        param.x_grid, param.y_grid, model.max_deg); 
    for i = 1:model.nDims
        for j = 1:model.nDims
            fits.Sigmas_recover(:,:,i,j) = sum(fits.U_recover(:,:,i,:).*...
                fits.U_recover(:,:,j,:),4);
        end
    end
    
    %% save the data
    E = {param, stim, results, plt, sim, model, fits};
    
    if strcmp(sim.method_sampling, 'NearContour')
        fileName = ['Fits_isothreshold_',plt.ttl{sim.slc_RGBplane},'_sim',...
            num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
            '_jitter',num2str(sim.random_jitter),'_rng',...
            num2str(s),'.mat'];
    elseif strcmp(sim.method_sampling, 'Random')
        fileName = ['Fits_isothreshold_',plt.ttl{sim.slc_RGBplane},'_sim',...
            num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
            '_range',num2str(sim.range_randomSampling(end)),...
            '_rng', num2str(s), '.mat'];
    end
    save(fileName,'E');

    %% clear data
    clear sim fits
end