clear all; close all; clc; rng(1)
% https://colab.research.google.com/drive/1oJGaDMElkiBkWgr3X0DkBf7Cr41pa8sX?usp=sharing#scrollTo=eGSlr-WH8xrA
%define some parameters
MAX_DEGREE     = 3;     %corresponds to p in Alex's document
NUM_DIMS       = 2;     %corresponds to a = 1, a = 2 
EXTRA_DIMS     = 0;
NUM_GRID_PTS   = 28;
NUM_MC_SAMPLES = 100;
DECAY_RATE     = 0.4;
VARIANCE_SCALE = 0.004;

%% make it 2d
%[T(0)T(0), T(0)T(1), T(0)T(2);
% T(1)T(0), T(1)T(1), T(1)T(2);
% T(2)T(0), T(2)T(1), T(2)T(2)]; 

%get the polynomial coefficients of the basis function
%T(0) = 1               -> [0, 0, 1] corresponds to coefficients to [x^2, x^1, x^0]
%T(1) = x               -> [0, 1, 0]
%T(2) = 2x^2 - 1        -> [2, 0,-1]

coeffs_chebyshev = compute_chebyshev_basis_coeffs(MAX_DEGREE);
disp(coeffs_chebyshev)

%% PART2: Sampling from the Wishart Process prior
%Sample W. As the degree of polynomial increases, the std of weights
%decreases. 
flag_plot   = false;
nInstances  = 10; 
NUM_TRIALS  = 100; 
x_sim_range = [-1, 1];
[XT, YT]   = meshgrid(linspace(-1,1, NUM_GRID_PTS), linspace(-1,1, NUM_GRID_PTS));
[W_true, U_true, Phi, Sigmas_true, etas, x_sim, xbar_sim, pCorrect_sim, resp_sim] = ...
    deal(cell(1,nInstances));
for n = 1:nInstances
    W_true{n} = sample_W_prior(MAX_DEGREE, NUM_DIMS, EXTRA_DIMS, ...
        VARIANCE_SCALE, DECAY_RATE);
    if flag_plot; plot_multiHeatmap(W_true{n},'permute_M',true); end

    %Compute U, which is essentially the weighted sum of basis functions
    [U_true{n},Phi{n}] = compute_U(coeffs_chebyshev, W_true{n}, XT, YT, MAX_DEGREE);

    % plot_multiHeatmap(Phi{n},'permute_M',true)
    % plot_multiHeatmap(U_true{n},'permute_M',true)

    Sigmas_true_ij = NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS);
    for i = 1:NUM_DIMS
        for j = 1:NUM_DIMS
            %size of U_true: NUM_GRID_PTS x NUM_GRID_PTS x NUM_DIMS x (NUM_DIMS + EXTRA_DIMS)
            Sigmas_true_ij(:,:,i,j) = sum(U_true{n}(:,:,i,:).*U_true{n}(:,:,j,:),4);
        end
    end
    Sigmas_true{n} = Sigmas_true_ij;

    % visualize it
    if flag_plot; plot_Sigma(Sigmas_true{n}, XT, YT); end

    % Simulating data
    %first simulate 100*100 pairs of reference and comparison stimuli
    etas{n}        = randn([2,1,NUM_DIMS + EXTRA_DIMS, NUM_MC_SAMPLES]); %NUM_MC_SAMPLES
    x_sim{n}       = rand([NUM_TRIALS,NUM_TRIALS, NUM_DIMS]).*diff(x_sim_range) + x_sim_range(1);
    xbar_sim_temp  = x_sim{n} + 0.1.*randn(NUM_TRIALS,NUM_TRIALS, NUM_DIMS);
    xbar_sim_temp(xbar_sim_temp <-1) = -1; xbar_sim_temp(xbar_sim_temp >1) = 1;
    xbar_sim{n} = xbar_sim_temp;
    
    %compute the predicted percent correct
    pCorrect_sim{n} = 1- predict_error_prob(W_true{n}, coeffs_chebyshev, x_sim{n}, xbar_sim{n}, ...
                'etas',etas{n});
    resp_sim{n} = binornd(1, pCorrect_sim{n}(:),size(pCorrect_sim{n}(:)));
end

%% save data
simData = struct('MAX_DEGREE',MAX_DEGREE,'NUM_DIMS',NUM_DIMS,...
    'EXTRA_DIMS',EXTRA_DIMS,'NUM_GRID_PTS',NUM_GRID_PTS,...
    'NUM_MC_SAMPLES',NUM_MC_SAMPLES,'NUM_TRIALS',NUM_TRIALS,...
    'coeffs_chebyshev',coeffs_chebyshev,'x_sim_range',x_sim_range,...
    'XT', XT,'YT',YT,'W_true',W_true, 'U_true',U_true, 'Phi',Phi, ...
    'Sigmas_true',Sigmas_true, 'etas',etas, 'x_sim',x_sim, ...
    'xbar_sim',xbar_sim, 'pCorrect_sim',pCorrect_sim,'resp_sim',resp_sim);
save('simData_parameterRecovery.mat','simData');

%% local plotting function
function plot_Sigma(Sigma, X, Y)
    thetas = linspace(0,2*pi, 50);
    sinusoids = [cos(thetas); sin(thetas)];
    figure
    for i = 1:2:size(X,1)
        for j = 1:2:size(X,2)
            sig_ij = sqrtm(squeeze(Sigma(i,j,:,:)))*sinusoids;
            plot(sig_ij(1,:)+X(i,j), sig_ij(2,:)+Y(i,j),'k'); hold on
        end
    end
    xlim([-1.5,1.5]);ylim([-1.5,1.5]); axis square;
    title('Sample from Wishart process prior');
end

