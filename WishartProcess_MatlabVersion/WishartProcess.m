clear all; close all; clc; 
% https://colab.research.google.com/drive/1oJGaDMElkiBkWgr3X0DkBf7Cr41pa8sX?usp=sharing#scrollTo=eGSlr-WH8xrA
%define some parameters

MAX_DEGREE     = 5;     %corresponds to p in Alex's document
NUM_DIMS       = 2;     %corresponds to a = 1, a = 2 
EXTRA_DIMS     = 1;
NUM_GRID_PTS   = 28;
NUM_MC_SAMPLES = 100;   %number of monte carlo simulation
DECAY_RATE     = 0.4;
VARIANCE_SCALE = 0.004;

%% Part 1: Wishart Process prior and approximation in Chebyshev polynomial basis 
% Chebyshev polynomials of the first kind
%T(0) = 1
%T(1) = x
%T(2) = 2x^2 - 1
%T(3) = 4x^3 - 3x
%T(4) = 8x^4 - 8x^2 + 1
%T(n) = 2xT(n-1) - T(n-2)
syms x
T  = chebyshevT(0:(MAX_DEGREE-1), x);
disp(T)

%visualize the basis functions
plot_chebyshevPolynomials(T)

%test for orthogonality
flag_nonZeros = 0;
for d1 = 1:(MAX_DEGREE-1)
    T_d1 = matlabFunction(T(d1));
    for d2 = (d1+1):MAX_DEGREE
        T_d2 = matlabFunction(T(d2));
        wfunc = @(x) 1./sqrt(1-x.^2);
        if d1 > 1; integrad = @(x) T_d1(x).*T_d2(x).*wfunc(x);
        else; integrad = @(x) T_d2(x).*wfunc(x);end
        integral_result = integral(integrad, -1,1);
        if abs(integral_result) > 1e-10; flag_nonZeros = 1; end
    end
end
if flag_nonZeros == 0; disp('Orthogonality checked!'); end


%% make it 2d
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
coeffs_chebyshev = compute_chebyshev_basis_coeffs(MAX_DEGREE);
disp(coeffs_chebyshev)

xg       = linspace(-1,1,NUM_MC_SAMPLES);
yg       = linspace(-1,1,NUM_MC_SAMPLES);
[XG, YG] = meshgrid(xg, yg);
XYG = cat(3, XG, YG);
[~, M_chebyshev] = compute_U(coeffs_chebyshev,[],XYG, MAX_DEGREE);

%visualize it
plot_multiHeatmap(M_chebyshev,'permute_M',true,'figPos',[0,0.1,0.4,0.7],...
    'figName','BasisFunctions_deg5','saveFig',false);

%% randomly generate linear combinations of the basis function
%try 4 instances
nInstances = 4; 
%draw values from Gaussian distributions and use them as weights for the
%basis functions
w = randn(MAX_DEGREE,MAX_DEGREE,2, 2).*0.01;
U_rand = compute_U(coeffs_chebyshev, w, XYG, MAX_DEGREE);

%visualize it
plot_multiHeatmap(U_rand,'permute_M',true)

%% PART2: Sampling from the Wishart Process prior
%Sample W. As the degree of polynomial increases, the std of weights
%decreases. 
% W_true = sample_W_prior(MAX_DEGREE, NUM_DIMS, EXTRA_DIMS, VARIANCE_SCALE, DECAY_RATE);

%For debugging, load W_true.m exported from Alex's code. This way, we can
%test if the code below works as expected.
clear W_true                 %comment it out after debugging
load('W_true.mat','W_true')
plot_multiHeatmap(W_true,'permute_M',true,'cmap',"sky",'saveFig',false,'figName','W_example'); 

%Compute U, which is essentially the weighted sum of basis functions
[XT, YT] = meshgrid(linspace(-1,1, NUM_GRID_PTS), linspace(-1,1, NUM_GRID_PTS));
XYT = cat(3, XT, YT);
[U_true,Phi] = compute_U(coeffs_chebyshev, W_true, XYT, MAX_DEGREE,'scalePhi_toRGB',false);

%Sanity check: if I didn't mess up with dimensions, then the following
%figure should look identical to the ones generated before (except the
%sampling is coarser) 
%visualize it
plot_multiHeatmap(Phi,'permute_M',true)

%also visualize U. It shouldn't be that much different than the ones 
%randomly generated before with randomly sampled weights
plot_multiHeatmap(U_true,'permute_M',true,'saveFig',false,'figName','U_example');

%% At the end, compute Sigmas_true given U
%the following code is equivalent to 
%Sigmas_true = jnp.einsum("ijdv,ijev->ijde", U_true, U_true) in python
%but in MATLAB tensorprod.m doesn't allow that operation, so I had to do it
%in a tedious way
Sigmas_true = NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS);
for i = 1:NUM_DIMS
    for j = 1:NUM_DIMS
        %size of U_true: NUM_GRID_PTS x NUM_GRID_PTS x NUM_DIMS x (NUM_DIMS + EXTRA_DIMS)
        Sigmas_true(:,:,i,j) = sum(U_true(:,:,i,:).*U_true(:,:,j,:),4);
        % Sigmas_rand(:,:,i,j) = sum(U_rand(:,:,i,:).*U_rand(:,:,j,:),4); 
    end
end

% visualize it
plot_Sigma(Sigmas_true, XT, YT)

% plot_Sigma(Sigmas_rand, XT, YT)
%Questions so far:
%1. what does the extra dimension represent? 

%% Part 3: Predicting the probability of error from the model
% %simulate eta
% etas = randn([2,1,NUM_DIMS + EXTRA_DIMS, NUM_MC_SAMPLES]); %NUM_MC_SAMPLES

%for debugging purpose, load etas.mat exported from Alex's code see if I
%can get exactly the same answer
load('etas.mat', 'etas')
%test a very simple example: x = 0 and xbar = 1
predict_error_prob(W_true, coeffs_chebyshev, zeros(1,1,NUM_DIMS), ...
    ones(1,1,NUM_DIMS), 'etas',etas,'scalePhi_toRGB',false)

%% define the property of comparison stimuli
XBAR        = NaN(NUM_GRID_PTS, NUM_GRID_PTS, 2);
XBAR(:,:,1) = XT; 
XBAR(:,:,2) = YT;

nX = 4;
[X1,X2] = meshgrid(linspace(-1,1,nX), linspace(-1,1,nX));
pIncorrect = NaN(nX,nX, NUM_GRID_PTS, NUM_GRID_PTS);
for a = 1:nX
    for b = 1:nX
        %define the property of reference stimuli 
        X = NaN(size(XT,1),size(XT,2),2);
        X(:,:,1) = X1(a,b).*ones(size(XT)); 
        X(:,:,2) = X2(a,b).*ones(size(XT));
        pIncorrect(a,b,:,:) = predict_error_prob(W_true, coeffs_chebyshev, X, XBAR, ...
            'etas',etas,'scalePhi_toRGB',false);
    end
end

% visualize it
plot_multiHeatmap(pIncorrect, 'X',linspace(-1,1,NUM_GRID_PTS),...
    'Y',linspace(-1,1,NUM_GRID_PTS),'D',cat(3,X1,X2), 'cmap', "sky",...
    'sgttl',"Error probability (relative to megenta star)");

%% Part 4a: Simulating data
%first simulate 50*50 pairs of reference and comparison stimuli
NUM_TRIALS  = 50; 
x_sim_range = [-1, 1];
x_sim       = rand([NUM_TRIALS,NUM_TRIALS, NUM_DIMS]).*diff(x_sim_range) + x_sim_range(1);
xbar_sim    = rand([NUM_TRIALS,NUM_TRIALS, NUM_DIMS]).*diff(x_sim_range) + x_sim_range(1);
% xbar_sim    = x_sim + 0.2.*randn(NUM_TRIALS,NUM_TRIALS, NUM_DIMS);
% xbar_sim(xbar_sim <-1) = -1; xbar_sim(xbar_sim >1) = 1;
%compute the predicted percent correct
pCorrect_sim = 1- predict_error_prob(W_true, coeffs_chebyshev, x_sim, xbar_sim, ...
            'etas',etas,'scalePhi_toRGB',false);
resp_sim = binornd(1, pCorrect_sim(:),size(pCorrect_sim(:)));
fprintf('Num correct trials: %.d\n', sum(resp_sim));
fprintf('Num error trials: %.d\n', NUM_TRIALS*NUM_TRIALS - sum(resp_sim));

%% Part 4b: Fitting the model to the data
w_reshape_size = [MAX_DEGREE, MAX_DEGREE, NUM_DIMS,NUM_DIMS+EXTRA_DIMS];
num_free_param_W = prod(w_reshape_size);
objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec,w_reshape_size,...
    x_sim, xbar_sim, resp_sim, coeffs_chebyshev,'etas', etas,'scalePhi_toRGB',false);

%have different initial points to avoid fmincon from getting stuck at
%some places
lb      = -1.*ones(1, num_free_param_W);
ub      = ones(1, num_free_param_W);
N_runs  = 2;
init    = rand(N_runs,num_free_param_W).*(ub-lb) + lb;
%fix some values
W_true_colvec        = W_true(:);
param_idx            = [1:2];%[6:9, 12:14,17:19];
nonParam_idx         = setdiff(1:length(W_true_colvec),param_idx); 
lb(nonParam_idx)     = W_true_colvec(nonParam_idx);
ub(nonParam_idx)     = W_true_colvec(nonParam_idx);
init(:,nonParam_idx) = repmat(W_true_colvec(nonParam_idx)',[N_runs,1]);

options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display','off');
w_colvec_est = NaN(N_runs, num_free_param_W);
minVal       = NaN(1, N_runs);
for n = 1:N_runs
    disp(n)
    %use fmincon to search for the optimal defocus
    [w_colvec_est(n,:), minVal(n)] = fmincon(objectiveFunc, init(n,:), ...
        [],[],[],[],lb,ub,[],options);
end
%find the index that corresponds to the minimum value
[~,idx_min] = min(minVal);
%find the corresponding optimal focus that leads to the highest peak of
%the psf's
w_colvec_est_best  = w_colvec_est(idx_min,:);
w_est_best = reshape(w_colvec_est_best, w_reshape_size);

%% Recover
[U_recover,Phi_recover] = compute_U(coeffs_chebyshev, w_est_best, XYT, MAX_DEGREE);
Sigmas_recover = NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS);
for i = 1:NUM_DIMS
    for j = 1:NUM_DIMS
        Sigmas_recover(:,:,i,j) = sum(U_recover(:,:,i,:).*U_recover(:,:,j,:),4);
    end
end

%visualize it
plot_Sigma(Sigmas_true, XT, YT)
plot_Sigma(Sigmas_recover, XT, YT)

%% local plotting function
function plot_chebyshevPolynomials(T)
    figure
    fplot(T,'LineWidth',2); axis([-1, 1, -1, 1]);
    ylabel('T_n(x)')
    legend('T_0(x)','T_1(x)','T_2(x)','T_3(x)','T_4(x)','Location','Best')
    title('Chebyshev polynomials of the first kind')
    %output folder names
    analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
    myFigDir = 'WishartPractice_FigFiles';
    outputDir = fullfile(analysisDir, myFigDir);
    figFilePath = fullfile(outputDir, 'ChebyshevPolynomials.pdf');
    % saveas(gcf, figFilePath);
end

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
    axis equal; xlim([-1.5,1.5]); ylim([-1.5,1.5]);set(gca,'FontSize',12)
    % title('Sample from Wishart process prior');
    %output folder names
    analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
    myFigDir = 'WishartPractice_FigFiles';
    outputDir = fullfile(analysisDir, myFigDir);
    figFilePath = fullfile(outputDir, 'Sigma_example.pdf');
    saveas(gcf, figFilePath);
end




