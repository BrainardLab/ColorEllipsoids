clear all; close all; clc

%load data
load('simData_parameterRecovery.mat', 'simData')
slc_simData      = 1;
D                = simData(slc_simData);
MAX_DEGREE       = D.MAX_DEGREE;
NUM_DIMS         = D.NUM_DIMS;
EXTRA_DIMS       = D.EXTRA_DIMS;
coeffs_chebyshev = D.coeffs_chebyshev;
x_sim            = D.x_sim;
xbar_sim         = D.xbar_sim;
etas             = D.etas;
resp_sim         = D.resp_sim;

%% Part 4b: Fitting the model to the data
w_reshape_size = [MAX_DEGREE, MAX_DEGREE, NUM_DIMS,NUM_DIMS+EXTRA_DIMS];
num_free_param_W = prod(w_reshape_size);
objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec,w_reshape_size,...
    x_sim, xbar_sim, resp_sim, coeffs_chebyshev, 'scalePhi_toRGB',false);

%have different initial points to avoid fmincon from getting stuck at
%some places
lb      = -0.2.*ones(1, num_free_param_W);
ub      = 0.2.*ones(1, num_free_param_W);
plb     = -0.1.*ones(1, num_free_param_W);
pub     = 0.1.*ones(1, num_free_param_W);
N_runs  = 1;
init    = rand(N_runs,num_free_param_W).*(ub-lb) + lb;

w_colvec_est = NaN(N_runs, num_free_param_W);
minVal       = NaN(1, N_runs);
for n = 1:N_runs
    disp(n)
    %use bads to search for the optimal defocus
    [w_colvec_est(n,:), minVal(n)] = bads(objectiveFunc,...
        init(n,:), lb, ub, plb, pub); 
end
%find the index that corresponds to the minimum value
[~,idx_min] = min(minVal);
%find the corresponding optimal focus that leads to the highest peak of
%the psf's
w_colvec_est_best  = w_colvec_est(idx_min,:);
w_est_best = reshape(w_colvec_est_best, w_reshape_size);

%% Recover
NUM_GRID_PTS = 28;
[XT, YT] = meshgrid(linspace(-0.8,0.8, NUM_GRID_PTS), linspace(-0.8,0.8, NUM_GRID_PTS));
[U_recover,Phi_recover] = compute_U(coeffs_chebyshev, w_est_best, XT, YT, MAX_DEGREE);
Sigmas_recover = NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS);
for i = 1:NUM_DIMS
    for j = 1:NUM_DIMS
        Sigmas_recover(:,:,i,j) = sum(U_recover(:,:,i,:).*U_recover(:,:,j,:),4);
    end
end

%visualize it
plot_Sigma(D.Sigmas_true, XT, YT)
plot_Sigma(Sigmas_recover, XT, YT)

%% local plotting function
function plot_Sigma(Sigma, X, Y)
    thetas = linspace(0,2*pi, 50);
    sinusoids = [cos(thetas); sin(thetas)];
    figure
    for i = 1:4:size(X,1)
        for j = 1:4:size(X,2)
            sig_ij = sqrtm(squeeze(Sigma(i,j,:,:)))*sinusoids;
            plot(sig_ij(1,:)+X(i,j), sig_ij(2,:)+Y(i,j),'k'); hold on
        end
    end
    xlim([-1.5,1.5]);ylim([-1.5,1.5]); axis square;
    title('Sample from Wishart process prior');
end