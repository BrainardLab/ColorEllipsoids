clear all; close all; clc; 
% https://colab.research.google.com/drive/1oJGaDMElkiBkWgr3X0DkBf7Cr41pa8sX?usp=sharing#scrollTo=eGSlr-WH8xrA
%define some parameters
global MAX_DEGREE NUM_DIMS EXTRA_DIMS NUM_GRID_PTS NUM_MC_SAMPLES
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
xg = linspace(-1,1,NUM_MC_SAMPLES);
yg = linspace(-1,1,NUM_MC_SAMPLES);
figure
fplot(T,'LineWidth',2); axis([-1, 1, -1, 1]);
ylabel('T_n(x)')
legend('T_0(x)','T_1(x)','T_2(x)','T_3(x)','T_4(x)','Location','Best')
title('Chebyshev polynomials of the first kind')

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
coeffs_chebyshev = compute_chebyshev_basis_coeffs;
disp(coeffs_chebyshev)
%%
[XG, YG]    = meshgrid(xg, yg);
[~, M_chebyshev] = compute_U(coeffs_chebyshev,[],XG,YG);

%visualize it
plot_heatmap(M_chebyshev)

%% randomly generate linear combinations of the basis function
%try 4 instances
nInstances = 4; 
%draw values from Gaussian distributions and use them as weights for the
%basis functions
w = randn(MAX_DEGREE,MAX_DEGREE,1, nInstances);
U_rand = compute_U(coeffs_chebyshev, w, XG, YG);

%visualize it
plot_heatmap(U_rand)

%% PART2: Sampling from the Wishart Process prior
%Sample W. As the degree of polynomial increases, the std of weights
%decreases. 
W_true = sample_W_prior(VARIANCE_SCALE, DECAY_RATE);

%For debugging, load W_true.m exported from Alex's code. This way, we can
%test if the code below works as expected.
clear W_true                 %comment it out after debugging
load('W_true.mat','W_true')
plot_heatmap(W_true)

%Compute U, which is essentially the weighted sum of basis functions
[XT, YT] = meshgrid(linspace(-1,1, NUM_GRID_PTS), linspace(-1,1, NUM_GRID_PTS));
[U_true,Phi] = compute_U(coeffs_chebyshev, W_true, XT, YT);

%Sanity check: if I didn't mess up with dimensions, then the following
%figure should look identical to the ones generated before (except the
%sampling is coarser) 
%visualize it
plot_heatmap(Phi)

%also visualize U. It shouldn't be that much different than the ones 
%randomly generated before with randomly sampled weights
plot_heatmap(U_true)

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
    end
end

%test for positive semi-definity
Sigmas_true_reshape = reshape(Sigmas_true, [NUM_GRID_PTS*NUM_GRID_PTS, NUM_DIMS, NUM_DIMS]);
nTest_pd = 0; 
for t = 1:NUM_GRID_PTS*NUM_GRID_PTS
    v_rand = rand(2,1);
    vSv = v_rand'*squeeze(Sigmas_true_reshape(t,:,:))*v_rand;
    if vSv >= 0; nTest_pd = nTest_pd + 1; end    
end
assert(nTest_pd==NUM_GRID_PTS*NUM_GRID_PTS,'Sigma is not positive semi-definite!');

% visualize it
plot_Sigma(Sigmas_true, XT, YT)

%Questions so far:
%1. what does the extra dimension represent? 
%2. the scaler for variance is not expressed in the function of phi
%3. when plotting samples from wishart process prior, why do we take a 
%   matrix square root of Sigmas_true

%% Part 3: Predicting the probability of error from the model
% %simulate eta
etas = randn([2,1,NUM_DIMS + EXTRA_DIMS, NUM_MC_SAMPLES]);

%for debugging purpose, load etas.mat exported from Alex's code see if I
%can get exactly the same answer
load('etas.mat', 'etas')
%test a very simple example: x = 0 and xbar = 1
predict_error_prob(zeros(1,1,NUM_DIMS), ones(1,1,NUM_DIMS), coeffs_chebyshev, W_true, etas)

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
        pIncorrect(a,b,:,:) = predict_error_prob(X, XBAR, coeffs_chebyshev,...
            W_true, etas);
    end
end

% visualize it
figure
colormap(sky)
for a = 1:nX
    for b = 1:nX
        subplot(nX,nX,b+nX*(a-1)); 
        imagesc(linspace(-1,1,NUM_GRID_PTS),linspace(-1,1,NUM_GRID_PTS),...
            squeeze(pIncorrect(a,b,:,:))); 
        axis square; hold on; xticks([]); yticks([])
        scatter(X1(a,b), X2(a,b),15,'mo','filled');
    end
end
sgtitle("Error probability (relative to megenta star)")

%% Part 4a: Simulating data
%first simulate 50*50 pairs of reference and comparison stimuli
NUM_TRIALS  = 50; 
x_sim_range = [-1, 1];
x_sim       = rand([NUM_TRIALS,NUM_TRIALS, NUM_DIMS]).*diff(x_sim_range) + x_sim_range(1);
xbar_sim    = rand([NUM_TRIALS,NUM_TRIALS, NUM_DIMS]).*diff(x_sim_range) + x_sim_range(1);
% xbar_sim    = x_sim + 0.2.*randn(NUM_TRIALS,NUM_TRIALS, NUM_DIMS);
% xbar_sim(xbar_sim <-1) = -1; xbar_sim(xbar_sim >1) = 1;
%compute the predicted percent correct
pCorrect_sim = 1-predict_error_prob(x_sim, xbar_sim, coeffs_chebyshev,...
            W_true, etas);
resp_sim = binornd(1, pCorrect_sim(:),size(pCorrect_sim(:)));
fprintf('Num correct trials: %.d\n', sum(resp_sim));
fprintf('Num error trials: %.d\n', NUM_TRIALS*NUM_TRIALS - sum(resp_sim));

%% Part 4b: Fitting the model to the data
objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec, x_sim, ...
    xbar_sim, resp_sim, coeffs_chebyshev, etas);

%have different initial points to avoid fmincon from getting stuck at
%some places
lb      = -1.*ones(1, MAX_DEGREE*MAX_DEGREE*NUM_DIMS*(NUM_DIMS+EXTRA_DIMS));
ub      = ones(1, MAX_DEGREE*MAX_DEGREE*NUM_DIMS*(NUM_DIMS+EXTRA_DIMS));
N_runs  = 2;
init    = rand(N_runs,MAX_DEGREE*MAX_DEGREE*NUM_DIMS*(NUM_DIMS+EXTRA_DIMS)).*(ub-lb) + lb;
%fix some values
W_true_colvec        = W_true(:);
param_idx            = [1:5];%[6:9, 12:14,17:19];
nonParam_idx         = setdiff(1:length(W_true_colvec),param_idx); 
lb(nonParam_idx)     = W_true_colvec(nonParam_idx);
ub(nonParam_idx)     = W_true_colvec(nonParam_idx);
init(:,nonParam_idx) = repmat(W_true_colvec(nonParam_idx)',[N_runs,1]);

options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display','off');
w_colvec_est = NaN(N_runs, MAX_DEGREE*MAX_DEGREE*NUM_DIMS*(NUM_DIMS+EXTRA_DIMS));
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
w_est_best = reshape(w_colvec_est_best, [MAX_DEGREE, MAX_DEGREE, NUM_DIMS, NUM_DIMS+EXTRA_DIMS]);

%% Recover
[U_recover,Phi_recover] = compute_U(coeffs_chebyshev, w_est_best, XT, YT);
Sigmas_recover = NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS);
for i = 1:NUM_DIMS
    for j = 1:NUM_DIMS
        Sigmas_recover(:,:,i,j) = sum(U_recover(:,:,i,:).*U_recover(:,:,j,:),4);
    end
end

%visualize it
plot_Sigma(Sigmas_true, XT, YT)
plot_Sigma(Sigmas_recover, XT, YT)

%% helping function
function coeffs = compute_chebyshev_basis_coeffs
    global MAX_DEGREE
    syms x
    T = chebyshevT(0:MAX_DEGREE-1, x);
    coeffs = zeros(MAX_DEGREE, MAX_DEGREE);
    for p = 1:MAX_DEGREE
        coeffs_p = sym2poly(T(p));
        coeffs(p,(MAX_DEGREE-length(coeffs_p)+1):end) = coeffs_p;
    end
end

function W = sample_W_prior(var_scale, decay_rate)
    global MAX_DEGREE NUM_DIMS EXTRA_DIMS 
    degs = repmat(0:(MAX_DEGREE-1),[MAX_DEGREE,1]) +...
        repmat((0:(MAX_DEGREE-1))',[1,MAX_DEGREE]);
    vars = var_scale.*(decay_rate.^degs);
    stds = sqrt(vars);
    %visualize it
    %plot_heatmap(stds);colorbar
    W = repmat(stds,[1,1,NUM_DIMS, NUM_DIMS+EXTRA_DIMS]).*...
        randn(MAX_DEGREE, MAX_DEGREE, NUM_DIMS, NUM_DIMS+EXTRA_DIMS);
end

function [U, phi] = compute_U(poly_chebyshev, W,xt,yt)
    global MAX_DEGREE 
    NUM_GRID_PTS1 = size(xt,1);
    NUM_GRID_PTS2 = size(xt,2);
    NUM_DIMS     = size(W,3);
    EXTRA_DIMS   = size(W,4)-NUM_DIMS;
    [val_xt, val_yt] = deal(NaN(NUM_GRID_PTS1, NUM_GRID_PTS2, MAX_DEGREE));
    for d = 1:MAX_DEGREE 
        val_xt(:,:,d) = polyval(poly_chebyshev(d,:),xt);
        val_yt(:,:,d) = polyval(poly_chebyshev(d,:),yt);
    end

    val_xt_repmat = repmat(val_xt, [1,1,1,size(poly_chebyshev,2)]);
    val_yt_repmat = permute(repmat(val_yt, [1,1,1,size(poly_chebyshev,2)]),[1,2,4,3]);
    phi = val_xt_repmat.*val_yt_repmat;
    % %visualize it
    % plot_heatmap(val_xt_repmat)
    % plot_heatmap(val_yt_repmat)
    % plot_heatmap(phi)
    % plot_heatmap(W)
    
    %equivalent of np.einsum(ij,jk-ik',A,B)
    %size of phi: NUM_GRID_PTS x NUM_GRID_PTS x MAX_DEGREE x MAX_DEGREE
    %size of W:   MAX_DEGREE   x MAX_DEGREE   x NUM_DIMS   x (NUM_DIMS+EXTRA_DIMS)
    if ~isempty(W)
        U = tensorprod(phi,W,[3,4],[1,2]); %w is eta in some of the equations
        % U = NaN(NUM_GRID_PTS,NUM_GRID_PTS,NUM_DIMS,NUM_DIMS+EXTRA_DIMS);
        % for i = 1:NUM_GRID_PTS
        %     for j = 1:NUM_GRID_PTS
        %         for k = 1:NUM_DIMS
        %             for l = 1:(NUM_DIMS+EXTRA_DIMS)
        %                 U(i,j,k,l) = sum(sum(squeeze(phi(i,j,:,:)).*squeeze(W(:,:,k,l))));
        %             end
        %         end
        %     end
        % end
    else
        U = [];
    end
end

function pIncorrect = predict_error_prob(x, xbar, coeffs_chebyshev,W_true, etas)
    global NUM_DIMS EXTRA_DIMS NUM_MC_SAMPLES
    NUM_GRID_PTS1   = size(x,1);
    NUM_GRID_PTS2   = size(x,2);
    %compute Sigma for the reference and the comparison stimuli
    U              = compute_U(coeffs_chebyshev,W_true, x(:,:,1), x(:,:,2)); 
    Ubar           = compute_U(coeffs_chebyshev,W_true, xbar(:,:,1), xbar(:,:,2)); 
    [Sigma, Sigma_bar] = deal(NaN(NUM_GRID_PTS1, NUM_GRID_PTS2, NUM_DIMS, NUM_DIMS));
    for i = 1:NUM_DIMS
        for j = 1:NUM_DIMS
            Sigma(:,:,i,j) = sum(U(:,:,i,:).*U(:,:,j,:),4); 
            Sigma_bar(:,:,i,j) = sum(Ubar(:,:,i,:).*Ubar(:,:,j,:),4); 
        end
    end

    %simulate values for the reference and the comparison stimuli
    etas_s       = etas(1,:,:,:); 
    etas_s       = reshape(etas_s, [1, NUM_DIMS+EXTRA_DIMS,NUM_MC_SAMPLES,1]);
    %Q: why is the noise generated this way?
    %size of U: NUM_GRID_PTS x NUM_GRID_PTS x NUM_DIMS x (NUM_DIMS + EXTRA_DIMS)
    %size of etas_bar: 1 x NUM_DIMS X (NUM_DIMS + EXTRA_DIMS)
    z_s_noise    = tensorprod(U, squeeze(etas_s), 4, 1);
    z_s          = repmat(x,[1,1,1,NUM_MC_SAMPLES]) + z_s_noise; 
    z_s          = permute(z_s,[1,2,4,3]);

    etas_sbar    = etas(2,:,:,:); 
    etas_sbar    = reshape(etas_sbar, [1,NUM_DIMS+EXTRA_DIMS,NUM_MC_SAMPLES,1]);
    z_sbar_noise = tensorprod(Ubar, squeeze(etas_sbar), 4, 1);
    z_sbar       = repmat(xbar,[1,1,1,NUM_MC_SAMPLES]) + z_sbar_noise;
    z_sbar       = permute(z_sbar,[1,2,4,3]);
    %visualize it
    % plot_heatmap(x); plot_heatmap(z_s_noise(:,:,:,1));
    % plot_heatmap(z_s(:,:,1,:));
    % plot_heatmap(xbar); plot_heatmap(z_sbar_noise(:,:,:,1));
    % plot_heatmap(z_sbar(:,:,1,:));

    %concatenate z_s and z_sbar
    z = NaN(NUM_GRID_PTS1, NUM_GRID_PTS2, 2*NUM_MC_SAMPLES, NUM_DIMS);
    z(:,:,1:NUM_MC_SAMPLES,:) = z_s;
    z(:,:,(NUM_MC_SAMPLES+1):end,:) = z_sbar;
    
    %compute prob
    pIncorrect = NaN(NUM_GRID_PTS1, NUM_GRID_PTS2);
    for t = 1:NUM_GRID_PTS1
        for v = 1:NUM_GRID_PTS2
            %predicting error probability
            p = mvnpdf(squeeze(z(t,v,:,:)), squeeze(x(t,v,:))', ...
                squeeze(Sigma(t,v,:,:)));
            q = mvnpdf(squeeze(z(t,v,:,:)), squeeze(xbar(t,v,:))', ...
                squeeze(Sigma_bar(t,v,:,:)));
            pIncorrect(t,v) = mean(min(p,q)./(p+q));
            
            % pC_x = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(x(t,v,:))',...
            %     squeeze(Sigma(t,v,:,:)));
            % pInc_x = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(x(t,v,:))',...
            %     squeeze(Sigma(t,v,:,:)));
            % pC_xbar = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(xbar(t,v,:))',...
            %     squeeze(Sigma_bar(t,v,:,:)));
            % pInc_xbar = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(xbar(t,v,:))',...
            %     squeeze(Sigma_bar(t,v,:,:)));
            % pC = pC_x.*pC_xbar;
            % pInc = pInc_x.*pInc_xbar;
            % p_normalization = pC + pInc;
            % pIncorrect(t,v) = mean(pInc./p_normalization);
        end
    end
end

function nLogL = estimate_loglikelihood(w_colvec, x, xbar, y, poly_chebyshev, etas)
    global MAX_DEGREE NUM_DIMS EXTRA_DIMS
    W = reshape(w_colvec, [MAX_DEGREE, MAX_DEGREE, NUM_DIMS,NUM_DIMS+ EXTRA_DIMS]);
    pInc = predict_error_prob(x, xbar, poly_chebyshev, W, etas);
    pC = 1 - pInc;
    logL = y.*log(pC(:)) + (1-y).*log(1-pInc(:));
    nLogL = -sum(logL(:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           PLOTING FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function plot_heatmap(M)
    nRows = size(M,3);
    nCols = size(M,4);
    figure
    for r = 1:nRows
        for c = 1:nCols
            colormap("summer")
            subplot(nRows, nCols, c+nCols*(r-1))
            imagesc(M(:,:,r,c));
            xticks([]); yticks([]); axis square;
        end
    end
end

function plot_Sigma(Sigma, X, Y)
    global NUM_GRID_PTS
    thetas = linspace(0,2*pi, 50);
    sinusoids = [cos(thetas); sin(thetas)];
    figure
    for i = 1:2:NUM_GRID_PTS
        for j = 1:2:NUM_GRID_PTS
            sig_ij = sqrtm(squeeze(Sigma(i,j,:,:)))*sinusoids;
            plot(sig_ij(1,:)+X(i,j), sig_ij(2,:)+Y(i,j),'k'); hold on
        end
    end
    title('Sample from Wishart process prior');
end




