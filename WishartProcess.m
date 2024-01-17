clear all; close all; clc; 
%define some parameters
MAX_DEGREE     = 5;
NUM_DIMS       = 2;
EXTRA_DIMS     = 1;
NUM_GRID_PTS   = 28;
DECAY_RATE     = 0.4;
VARIANCE_SCALE = 0.004;
NUM_MC_SAMPLES = 100;

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
[XG, YG]    = meshgrid(xg, yg);
M_chebyshev = NaN(MAX_DEGREE,MAX_DEGREE,NUM_MC_SAMPLES,NUM_MC_SAMPLES);
for i = 1:MAX_DEGREE
    %grab the function handle
    Ti_func = matlabFunction(T(i));
    %T(0) = @()1.0 
    %it doesn't take any input arguments, so let's make it a special case
    if i==1; Ti = ones(size(XG)); else; Ti = Ti_func(XG); end
    for j = 1:MAX_DEGREE
        Tj_func = matlabFunction(T(j));
        if j == 1; Tj = ones(size(YG)); else; Tj = Tj_func(YG); end
        M_chebyshev(i,j,:,:) = Ti.*Tj;
    end
end

%visualize it
figure
for i = 1:MAX_DEGREE
    for j = 1:MAX_DEGREE
        subplot(MAX_DEGREE,MAX_DEGREE,j+(i-1)*5); colormap(summer)
        imagesc(squeeze(M_chebyshev(i,j,:,:)));
        xticks([]); yticks([]); axis square;
    end
end

%% randomly generate linear combinations of the basis function
%try 4 instances
nInstances = 4;
%draw values from Gaussian distributions and use them as weights for the
%basis functions
w = randn(MAX_DEGREE,MAX_DEGREE,nInstances);
M_chebyshev_rand = NaN(nInstances,NUM_MC_SAMPLES,NUM_MC_SAMPLES);
for n = 1:nInstances
    %size of w_rep: MAX_DEGREE x MAX_DEGREE x NUM_MC_SAMPLES x NUM_MC_SAMPLES
    %               (5 x 5 x 100 x 100)
    w_rep = repmat(w(:,:,n),[1,1,NUM_MC_SAMPLES,NUM_MC_SAMPLES]);
    %sum all the 2d basis funcitons (across the 1st and 2nd dimensions)
    M_chebyshev_rand(n,:,:) = sum(sum(M_chebyshev.*w_rep,1),2);
end

figure
for n = 1:nInstances
    subplot(1,nInstances,n); colormap(summer)
    imagesc(squeeze(M_chebyshev_rand(n,:,:)));
    xticks([]); yticks([]); axis square;
end

%% PART2: Sampling from the Wishart Process prior
%get the polynomial coefficients of the basis function
%T(0) = 1               -> [0, 0, 0, 0, 1] corresponds to coefficients to [x^5, x^4, x^3, x^2, x^1, x^0]
%T(1) = x               -> [0, 0, 0, 1, 0]
%T(2) = 2x^2 - 1        -> [0, 0, 2, 0,-1]
%T(3) = 4x^3 - 3x       -> [0, 4, 0,-3, 0]
%T(4) = 8x^4 - 8x^2 + 1 -> [8, 0,-8, 0, 1]
Basis_chebyshev = compute_chebyshev_basis_coeffs(MAX_DEGREE);
disp(Basis_chebyshev)

%Sample W. As the degree of polynomial increases, the std of weights
%decreases. 
[W_true, logprior_W] = sample_W_prior(MAX_DEGREE, NUM_DIMS, EXTRA_DIMS,...
    VARIANCE_SCALE, DECAY_RATE);

%For debugging, load W_true.m exported from Alex's code. This way, we can
%test if the code below works as expected.
clear W_true %comment it out after debugging
load('W_true.mat','W_true')

%Compute U, which is essentially the weighted sum of basis functions
[XT, XV] = meshgrid(linspace(-1,1, NUM_GRID_PTS), linspace(-1,1, NUM_GRID_PTS));
[U_true,Phi] = compute_U(Basis_chebyshev, W_true, XT, XV);
%Sanity check: if I didn't mess up with dimensions, then the following
%figure should look identical to the ones generated before (except the
%sampling is coarser) 
figure
for i = 1:MAX_DEGREE
    for j = 1:MAX_DEGREE
        subplot(MAX_DEGREE,MAX_DEGREE,j+(i-1)*5); colormap(summer)
        imagesc(squeeze(Phi(:,:,i,j)));
        xticks([]); yticks([]); axis square;
    end
end

%also visualize U. It shouldn't be that much different than the ones 
%randomly generated before with randomly sampled weights
figure
for i = 1:NUM_DIMS
    for j = 1:(NUM_DIMS+EXTRA_DIMS)
        subplot(NUM_DIMS, NUM_DIMS+EXTRA_DIMS, j+(i-1)*(NUM_DIMS+EXTRA_DIMS))
        colormap(summer)
        imagesc(squeeze(U_true(:,:,i,j))); 
        xticks([]); yticks([]); axis square;
    end
end

%% At the end, compute Sigmas_true given U
%the following code is equivalent to 
%Sigmas_true = jnp.einsum("ijdv,ijev->ijde", U_true, U_true) in python
%but in MATLAB tensorprod.m doesn't allow that operation, so I had to do it
%in a tedious way

% Reshape U for the operation, aligning the dimensions for multiplication
U1 = reshape(U_true, [], size(U_true, 3), size(U_true, 4));
U2 = reshape(U_true, [], size(U_true, 3), size(U_true, 4));

% Perform the tensor contraction
% Multiplying and summing over the last dimension of U1 and the fourth dimension of U2
C = zeros(size(U1, 1), size(U1, 2), size(U2, 2));
for i = 1:size(U1, 2)
    for j = 1:size(U2, 2)
        C(:, i, j) = sum(U1(:, i, :) .* U2(:, j, :), 3);
    end
end

% Reshape C back to the desired 4D format (28 x 28 x 2 x 2)
Sigmas_true = reshape(C, size(U_true, 1), size(U_true, 2),...
    size(U_true, 3), size(U_true, 3));

% visualize
thetas = linspace(0,2*pi, 50);
sinusoids = [cos(thetas); sin(thetas)];
figure
for i = 1:2:NUM_GRID_PTS
    for j = 1:2:NUM_GRID_PTS
        sig_ij = sqrtm(squeeze(Sigmas_true(i,j,:,:)))*sinusoids;
        plot(sig_ij(1,:)+XT(i,j), sig_ij(2,:)+XV(i,j),'k'); hold on
    end
end
title('Sample from Wishart process prior');

%% helping function
function coeffs = compute_chebyshev_basis_coeffs(max_deg)
    syms x
    T = chebyshevT(0:max_deg-1, x);
    coeffs = zeros(max_deg, max_deg);
    for p = 1:max_deg
        coeffs_p = sym2poly(T(p));
        coeffs(p,(max_deg-length(coeffs_p)+1):end) = coeffs_p;
    end
end

function [W, sum_logpriorW] = sample_W_prior(max_degree, num_dims,...
    extra_dims, var_scale, decay_rate)
    degs = repmat(0:(max_degree-1),[max_degree,1]) +...
        repmat((0:(max_degree-1))',[1,max_degree]);
    vars = var_scale.*(decay_rate.^degs);
    stds = sqrt(vars);
    W = repmat(stds,[1,1,num_dims, num_dims+extra_dims]).*...
        randn(max_degree, max_degree, num_dims, num_dims+extra_dims);
    logpriorW = lognpdf(W, repmat(stds,[1,1,num_dims, num_dims+extra_dims]));
    sum_logpriorW = sum(logpriorW(:));
end

function [U, phi] = compute_U(poly_chebyshev, W,xt,xv)
    [val_xt, val_xv] = deal(NaN(size(xt,1), size(xt,2),size(poly_chebyshev,1)));
    for d = 1:size(poly_chebyshev,1)
        val_xt(:,:,d) = polyval(poly_chebyshev(d,:),xt);
        val_xv(:,:,d) = polyval(poly_chebyshev(d,:),xv);
    end
    val_xxt = repmat(val_xt, [1,1,1,size(poly_chebyshev,2)]);
    val_xxv = permute(repmat(val_xv, [1,1,1,size(poly_chebyshev,2)]),[1,2,4,3]);
    % val_xxv = repmat(val_xv, [1,1,1,size(poly_chebyshev,2)]);

    %size of phi: 28 x 28 x 5 x 5
    %size of W: 5 x 5 x 2 x 3
    phi = val_xxt.*val_xxv;
    %equivalent of np.einsum(ij,jk-ik',A,B)
    U = tensorprod(phi,W,[3,4],[1,2]);
end






