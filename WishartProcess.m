clear all; close all; clc; 
%define some parameters
global MAX_DEGREE NUM_DIMS EXTRA_DIMS NUM_GRID_PTS NUM_MC_SAMPLES
MAX_DEGREE     = 5;     %corresponds to p in Alex's document
NUM_DIMS       = 2;     %corresponds to a = 1, a = 2 
EXTRA_DIMS     = 1;
NUM_GRID_PTS   = 28;
NUM_MC_SAMPLES = 100;
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
fplot(T); axis([-1, 1, -1, 1]);
ylabel('T_n(x)')
legend('T_0(x)','T_1(x)','T_2(x)','T_3(x)','T_4(x)','Location','Best')
title('Chebyshev polynomials of the first kind')

%% make it 2d
%[T(0)T(0), T(0)T(1), ... , T(0)T(4);
% T(1)T(0), T(1)T(1), ... , T(1)T(4);
% ...       ...       ...   ...
% T(4)T(0), T(4)T(1), ... , T(4)T(4)];

%get the polynomial coefficients of the basis function
%T(0) = 1               -> [0, 0, 0, 0, 1] corresponds to coefficients to [x^5, x^4, x^3, x^2, x^1, x^0]
%T(1) = x               -> [0, 0, 0, 1, 0]
%T(2) = 2x^2 - 1        -> [0, 0, 2, 0,-1]
%T(3) = 4x^3 - 3x       -> [0, 4, 0,-3, 0]
%T(4) = 8x^4 - 8x^2 + 1 -> [8, 0,-8, 0, 1]
Basis_chebyshev = compute_chebyshev_basis_coeffs;
disp(Basis_chebyshev)

[XG, YG]    = meshgrid(xg, yg);
[~, M_chebyshev] = compute_U(Basis_chebyshev,[],XG,YG);

%visualize it
plot_heatmap(M_chebyshev)

%% randomly generate linear combinations of the basis function
%try 4 instances
nInstances = 4;
%draw values from Gaussian distributions and use them as weights for the
%basis functions
w = randn(MAX_DEGREE,MAX_DEGREE,nInstances,1);
M_chebyshev_rand = compute_U(Basis_chebyshev, w, XG, YG);

%visualize it
plot_heatmap(M_chebyshev_rand)

%% PART2: Sampling from the Wishart Process prior
%Sample W. As the degree of polynomial increases, the std of weights
%decreases. 
[W_true, logprior_W] = sample_W_prior(VARIANCE_SCALE, DECAY_RATE);

%For debugging, load W_true.m exported from Alex's code. This way, we can
%test if the code below works as expected.
clear W_true                 %comment it out after debugging
load('W_true.mat','W_true')

%Compute U, which is essentially the weighted sum of basis functions
[XT, YT] = meshgrid(linspace(-1,1, NUM_GRID_PTS), linspace(-1,1, NUM_GRID_PTS));
[U_true,Phi] = compute_U(Basis_chebyshev, W_true, XT, YT);

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
        Sigmas_true(:,:,i,j) = sum(U_true(:,:,i,:).*U_true(:,:,j,:),4);
    end
end

% visualize
thetas = linspace(0,2*pi, 50);
sinusoids = [cos(thetas); sin(thetas)];
figure
for i = 1:2:NUM_GRID_PTS
    for j = 1:2:NUM_GRID_PTS
        sig_ij = sqrtm(squeeze(Sigmas_true(i,j,:,:)))*sinusoids;
        plot(sig_ij(1,:)+XT(i,j), sig_ij(2,:)+YT(i,j),'k'); hold on
    end
end
title('Sample from Wishart process prior');

%Questions so far:
%1. what does the extra dimension represent? 
%2. the scaler for variance is not expressed in the function of phi

%% Part 3: Predicting the probability of error from the model
% %simulate eta
% etas = randn([2,1,NUM_DIMS + EXTRA_DIMS, NUM_MC_SAMPLES]);

%for debugging purpose, load etas.mat exported from Alex's code see if I
%cna get exactly the same answer
load('etas.mat', 'etas')
%test a very simple example: x = 0 and xbar = 1
predict_error_prob(zeros(1,1,NUM_DIMS), ones(1,1,NUM_DIMS),...
    Basis_chebyshev, W_true, etas)

%let's set a grid for x and xbar
XBAR = NaN(NUM_GRID_PTS, NUM_GRID_PTS, 2);
XBAR(:,:,1) = XT; XBAR(:,:,2) = YT;
nX = 4;
[XVAL_1, XVAL_2] = meshgrid(linspace(-1,1,nX), linspace(-1,1,nX));
probs = NaN(nX,nX, NUM_GRID_PTS, NUM_GRID_PTS);
for a = 1:nX
    for b = 1:nX
        X = NaN(size(XT,1),size(XT,2),2);
        X(:,:,1) = XVAL_1(a,b).*ones(size(XT)); 
        X(:,:,2) = XVAL_2(a,b).*ones(size(XT));
        probs(a,b,:,:) = predict_error_prob(X, XBAR, Basis_chebyshev,...
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
            squeeze(probs(a,b,:,:))); 
        axis square; hold on; xticks([]); yticks([])
        scatter(XVAL_1(a,b), XVAL_2(a,b),15,'mo','filled');
    end
end
sgtitle("Error probability (relative to megenta star)")

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

function [W, sum_logpriorW] = sample_W_prior(var_scale, decay_rate)
    global MAX_DEGREE NUM_DIMS EXTRA_DIMS 
    degs = repmat(0:(MAX_DEGREE-1),[MAX_DEGREE,1]) +...
        repmat((0:(MAX_DEGREE-1))',[1,MAX_DEGREE]);
    vars = var_scale.*(decay_rate.^degs);
    stds = sqrt(vars);
    W = repmat(stds,[1,1,NUM_DIMS, NUM_DIMS+EXTRA_DIMS]).*...
        randn(MAX_DEGREE, MAX_DEGREE, NUM_DIMS, NUM_DIMS+EXTRA_DIMS);
    logpriorW = lognpdf(W, repmat(stds,[1,1,NUM_DIMS, NUM_DIMS+EXTRA_DIMS]));
    sum_logpriorW = sum(logpriorW(:));
end

function [U, phi] = compute_U(poly_chebyshev, W,xt,yt)
    global MAX_DEGREE 
    NUM_GRID_PTS = size(xt,1);
    [val_xt, val_yt] = deal(NaN(NUM_GRID_PTS, NUM_GRID_PTS, MAX_DEGREE));
    for d = 1:MAX_DEGREE 
        val_xt(:,:,d) = polyval(poly_chebyshev(d,:),xt);
        val_yt(:,:,d) = polyval(poly_chebyshev(d,:),yt);
    end

    val_xt_repmat = repmat(val_xt, [1,1,1,size(poly_chebyshev,2)]);
    val_yt_repmat = permute(repmat(val_yt, [1,1,1,size(poly_chebyshev,2)]),[1,2,4,3]);
    %visualize it
    % plot_heatmap(val_xt_repmat)
    % plot_heatmap(val_yt_repmat)

    %size of phi: NUM_GRID_PTS x NUM_GRID_PTS x MAX_DEGREE x MAX_DEGREE
    %size of W: MAX_DEGREE x MAX_DEGREE x NUM_DIMS x (NUM_DIMS+EXTRA_DIMS)
    phi = val_xt_repmat.*val_yt_repmat;
    
    %equivalent of np.einsum(ij,jk-ik',A,B)
    if ~isempty(W)
        U = tensorprod(phi,W,[3,4],[1,2]); %w is eta in some of the equations
    else
        U = [];
    end
end

function prob = predict_error_prob(x, xbar, Basis_chebyshev,W_true, etas)
    global NUM_DIMS EXTRA_DIMS NUM_MC_SAMPLES
    NUM_GRID_PTS   = size(x,1);
    etas_s         = etas(1,:,:,:); 
    etas_s         = reshape(etas_s, [1, NUM_DIMS+EXTRA_DIMS,NUM_MC_SAMPLES,1]);
    etas_sbar      = etas(2,:,:,:); 
    etas_sbar      = reshape(etas_sbar, [1,NUM_DIMS+EXTRA_DIMS,NUM_MC_SAMPLES,1]);
    U              = compute_U(Basis_chebyshev,W_true, x(:,:,1), x(:,:,2)); 
    Ubar           = compute_U(Basis_chebyshev,W_true, xbar(:,:,1), xbar(:,:,2)); 
    
    [S, Sbar] = deal(NaN(NUM_GRID_PTS, NUM_GRID_PTS, NUM_DIMS, NUM_DIMS));
    for i = 1:NUM_DIMS
        for j = 1:NUM_DIMS
            S(:,:,i,j) = sum(U(:,:,i,:).*U(:,:,j,:),4); 
            Sbar(:,:,i,j) = sum(Ubar(:,:,i,:).*Ubar(:,:,j,:),4); 
        end
    end

    z_s    = repmat(x,[1,1,1,NUM_MC_SAMPLES]) + tensorprod(U, squeeze(etas_s), 4, 1); 
    z_s    = permute(z_s,[1,2,4,3]);
    z_sbar = repmat(xbar,[1,1,1,NUM_MC_SAMPLES]) + tensorprod(Ubar, squeeze(etas_sbar), 4, 1);
    z_sbar = permute(z_sbar,[1,2,4,3]);
    z      = NaN(NUM_GRID_PTS, NUM_GRID_PTS, 2*NUM_MC_SAMPLES, NUM_DIMS);
    z(:,:,1:NUM_MC_SAMPLES,:) = z_s;
    z(:,:,(NUM_MC_SAMPLES+1):end,:) = z_sbar;
    
    %compute prob
    prob = NaN(NUM_GRID_PTS, NUM_GRID_PTS);
    for t = 1:NUM_GRID_PTS
        for v = 1:NUM_GRID_PTS
            p = mvnpdf(squeeze(z(t,v,:,:)), squeeze(x(t,v,:))', squeeze(S(t,v,:,:)));
            q = mvnpdf(squeeze(z(t,v,:,:)), squeeze(xbar(t,v,:))', squeeze(Sbar(t,v,:,:)));
            prob(t,v) = mean(min(p,q)./(p+q));
        end
    end
end

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




