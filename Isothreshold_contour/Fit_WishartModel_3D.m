clear all; close all; clc; 

%% load isothreshold contours simulated based on CIELAB
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myDataDir   = 'Simulation_DataFiles';
intendedDir = fullfile(analysisDir, myDataDir);
addpath(intendedDir);
load('Sims_Isothreshold_ellipsoids_GB plane_sim240perCond_samplingNearContour_jitter0.1.mat', 'sim')
load('Isothreshold_ellipsoid_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% define model specificity for the wishart process
model.max_deg           = 3;     %corresponds to p in Alex's document
model.nDims             = 3;     %corresponds to a = 1, a = 2 
model.eDims             = 0;     %extra dimensions
model.num_grid_pts      = 100;
model.num_MC_samples    = 100;   %number of monte carlo simulation

%[T(0)T(0), T(0)T(1), ... , T(0)T(4);
% T(1)T(0), T(1)T(1), ... , T(1)T(4);
% ...       ...       ...   ...
% T(4)T(0), T(4)T(1), ... , T(4)T(4)]; General formula: T(m)T(n), m -> row - 1, n -> col - 1

%get the polynomial coefficients of the basis function
%T(0) = 1               -> [0, 0, 0, 0, 1] corresponds to coefficients to [x^5, x^4, x^3, x^2, x^1, x^0]
%T(1) = x               -> [0, 0, 0, 1, 0]
%T(2) = 2x^2 - 1        -> [0, 0, 2, 0,-1]
%T(3) = 4x^3 - 3x       -> [0, 4, 0,-3, 0]
%T(4) = 8x^4 - 8x^2 + 1 -> [8, 0,-8, 0, 1]
model.coeffs_chebyshev = compute_chebyshev_basis_coeffs(model.max_deg);
disp(model.coeffs_chebyshev)

model.xt = linspace(-1,1,model.num_MC_samples);
model.yt = linspace(-1,1,model.num_MC_samples);
model.zt = linspace(-1,1,model.num_MC_samples);
[model.XT, model.YT, model.ZT]   = meshgrid(model.xt, model.yt, model.zt);
XYZT = cat(4, model.XT, model.YT, model.ZT);
[~, model.M_chebyshev] = compute_U(model.coeffs_chebyshev,[],...
    XYZT, model.max_deg, 'scalePhi_toRGB',true);

%visualize it
%since it's 3d, it's hard to visualize (but we can slice it)
%only pick the basis functions along the diagonal 
M_chebyshev_diagonal = NaN(model.num_MC_samples,model.num_MC_samples,...
    model.num_MC_samples,model.max_deg);
for d = 1:model.max_deg
    M_chebyshev_diagonal(:,:,:,d) = squeeze(model.M_chebyshev(:,:,:,d,d,d));
end
%slice along z-axis
slc_slice = [1,25,50,75,100];
M_chebyshev_reshape = NaN(model.num_MC_samples,model.num_MC_samples,...
    model.max_deg,length(slc_slice));
for s = 1:length(slc_slice)
    M_chebyshev_reshape(:,:,:,s) = squeeze(M_chebyshev_diagonal(:,:,slc_slice(s),:));
end
plot_multiHeatmap(M_chebyshev_reshape,'permute_M',true,'figPos',[0,0.1,0.4,0.4],...
    'figName','BasisFunctions_deg5','saveFig',false,'colorbar_on', true);

%% Fit the model to the data
%comparison stimulus
%x_sim: stim.nGridPts_ref x stim.nGridPts_ref x dims x sim.nSims

sim.slc_ref             = 1:stim.nGridPts_ref;
x_sim                   = sim.rgb_comp(sim.slc_ref,sim.slc_ref,sim.slc_ref,sim.varying_RGBplane,:);
x_sim_d1                = squeeze(x_sim(:,:,:,1,:));
x_sim_d2                = squeeze(x_sim(:,:,:,2,:));
x_sim_d3                = squeeze(x_sim(:,:,:,3,:));
sim.x1_sim_org(1,:,1)    = x_sim_d1(:); 
sim.x1_sim_org(1,:,2)    = x_sim_d2(:);
sim.x1_sim_org(1,:,3)    = x_sim_d3(:);

%reference stimulus
xbar_sim                = repmat(sim.ref_points(sim.slc_ref,sim.slc_ref,...
                            sim.slc_ref,sim.varying_RGBplane),...
                            [1,1,1,sim.nSims]);
xbar_sim_d1             = squeeze(xbar_sim(:,:,:,1,:));
xbar_sim_d2             = squeeze(xbar_sim(:,:,:,2,:));
xbar_sim_d3             = squeeze(xbar_sim(:,:,:,3,:));
sim.xref_sim_org(1,:,1) = xbar_sim_d1(:); 
sim.xref_sim_org(1,:,2) = xbar_sim_d2(:);
sim.xref_sim_org(1,:,3) = xbar_sim_d3(:);

sim.x0_sim_org = sim.xref_sim_org;

% if we just want to use one set of randomly drawn samples
% sim.etas = 0.01.*randn([2,1,model.nDims + model.eDims, model.num_MC_samples]);

%response
resp_binary = sim.resp_binary(sim.slc_ref, sim.slc_ref, sim.slc_ref,:);

%define objective functions
w_reshape_size = [model.max_deg, model.max_deg, model.max_deg, model.nDims, model.nDims+model.eDims];

disp(prod(w_reshape_size))
disp('That''s too many free parameters!');

