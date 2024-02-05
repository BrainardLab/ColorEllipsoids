clear all; close all; clc

%% load isothreshold contours simulated based on CIELAB
load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};

%% First create a cube and select the RG, the RB and the GB planes
sim.slc_fixedVal     = 0.5; %the level of the fixed plane
sim.slc_fixedVal_idx = find(stim.fixed_RGBvec == sim.slc_fixedVal);
sim.slc_RGBplane     = 1; %GB plane with a fixed R value
sim.plane_points     = param.plane_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.ref_points       = stim.ref_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.background_RGB   = stim.background_RGB(:,sim.slc_fixedVal_idx);

%% compute iso-threshold contour
sim.alpha    = 1.1729;
sim.beta     = 1.2286;
sim.nSims    = 1e3;

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_pij = squeeze(sim.ref_points(i,j,:));
        %convert it to Lab
        [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param, sim.background_RGB, rgb_ref_pij);
        sim.ref_Lab(i,j,:) = ref_Lab_lpij;
        
        %simulate comparison stimulus
        rgb_comp_sim    = rgb_ref_pij + 0.01.*randn(3, sim.nSims);
        rgb_comp_sim(1,:) = sim.slc_fixedVal;
        rgb_comp_sim(rgb_comp_sim <0) =0; 
        rgb_comp_sim(rgb_comp_sim >1) = 1;
        sim.rgb_comp(i,j,:,:) = rgb_comp_sim;
        
        %simulate binary responses
        for n = 1:sim.nSims
            [sim.lab_comp(i,j,:,n), ~, ~] = convert_rgb_lab(param, ...
                sim.background_RGB, rgb_comp_sim(:,n));
            sim.deltaE(i,j,n) = norm(squeeze(sim.lab_comp(i,j,:,n)) - ref_Lab_lpij);
            sim.probC(i,j,n) = ComputeWeibTAFC(sim.deltaE(i,j,n), ...
                sim.alpha, sim.beta);
        end
        sim.resp_binary(i,j,:) = binornd(1, squeeze(sim.probC(i,j,:)), [sim.nSims, 1]);
    end
end


% %% visualize the probability of correct
% figure
% for i = 1:stim.nGridPts_ref
%     for j = 1:stim.nGridPts_ref
%         subplot(stim.nGridPts_ref, stim.nGridPts_ref, j + (i-1)*stim.nGridPts_ref)
%         scatter(squeeze(sim.deltaE(i,j,:)), squeeze(sim.probC(i,j,:))); hold on
%     end
% end

%% fit wishart process
model.max_deg        = 3;     %corresponds to p in Alex's document
model.nDims          = 2;     %corresponds to a = 1, a = 2 
model.eDims          = 0; %extra dimensions
model.num_grid_pts   = 100;
model.num_MC_samples = 200;   %number of monte carlo simulation

model.coeffs_chebyshev = compute_chebyshev_basis_coeffs(model.max_deg);
disp(model.coeffs_chebyshev)

xg = linspace(-1,1,model.num_MC_samples);
yg = linspace(-1,1,model.num_MC_samples);
[XG, YG]    = meshgrid(xg, yg);
[~, M_chebyshev] = compute_U(coeffs_chebyshev,[],XG,YG);

%visualize it
plot_heatmap(M_chebyshev)

%% Part 4b: Fitting the model to the data
%rescale
x_sim = stim.rgb_comp_sim(:,:,2:3,:);
x_sim_d1 = squeeze(x_sim(:,:,1,:));
x_sim_d2 = squeeze(x_sim(:,:,2,:));
x_sim_org = NaN(1,nSims*5*5,2);
x_sim_org(1,:,1) = x_sim_d1(:); x_sim_org(1,:,2) = x_sim_d2(:);

xbar_sim = repmat(stim.ref_points(:,:,2:3),[1,1,1,nSims]);
xbar_sim_d1 = squeeze(xbar_sim(:,:,1,:));
xbar_sim_d2 = squeeze(xbar_sim(:,:,2,:));
xbar_sim_org = NaN(1,nSims*5*5,2);
xbar_sim_org(1,:,1) = xbar_sim_d1(:); xbar_sim_org(1,:,2) = xbar_sim_d2(:);

etas = 0.01.*randn([2,1,nDims + eDims, NUM_MC_SAMPLES]);

objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec, x_sim_org, ...
    xbar_sim_org, results.resp_sim(:), coeffs_chebyshev, etas);

%have different initial points to avoid fmincon from getting stuck at
%some places
lb      = -0.05.*ones(1, fits.max_deg*fits.max_deg*nDims*(nDims+eDims));
ub      = 0.05.*ones(1, fits.max_deg*fits.max_deg*nDims*(nDims+eDims));
plb     = -0.01.*ones(1, fits.max_deg*fits.max_deg*nDims*(nDims+eDims));
pub     = 0.01.*ones(1, fits.max_deg*fits.max_deg*nDims*(nDims+eDims));
N_runs  = 1;
init    = rand(N_runs,fits.max_deg*fits.max_deg*nDims*(nDims+eDims)).*(ub-lb) + lb;

options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display','iter-detailed');
w_colvec_est = NaN(N_runs, fits.max_deg*fits.max_deg*nDims*(nDims+eDims));
minVal       = NaN(1, N_runs);
for n = 1:N_runs
    disp(n)
    %use fmincon to search for the optimal defocus
    [w_colvec_est(n,:), minVal(n)] = bads(objectiveFunc, init(n,:),lb,ub,plb,pub);
    % [w_colvec_est(n,:), minVal(n)] = fmincon(objectiveFunc, init(n,:), ...
    %     [],[],[],[],lb,ub,[],options);
end
%find the index that corresponds to the minimum value
[~,idx_min] = min(minVal);
%find the corresponding optimal focus that leads to the highest peak of
%the psf's
w_colvec_est_best  = w_colvec_est(idx_min,:);
w_est_best = reshape(w_colvec_est_best, [fits.max_deg, fits.max_deg, nDims, nDims+eDims]);

%% save the data
D = {param, stim, results, M_chebyshev, w_est_best};
save("Fits_isothreshold_GBplane.mat","D");

%% recover
[XT, YT] = meshgrid(linspace(0,1, num_grid_pts), linspace(0,1, num_grid_pts));

[U_recover,Phi_recover] = compute_U(coeffs_chebyshev, w_est_best, XT, YT);
Sigmas_recover = NaN(num_grid_pts, num_grid_pts, nDims, nDims);
for i = 1:nDims
    for j = 1:nDims
        Sigmas_recover(:,:,i,j) = sum(U_recover(:,:,i,:).*U_recover(:,:,j,:),4);
    end
end

thetas = linspace(0,2*pi, 50);
sinusoids = [cos(thetas); sin(thetas)];
figure
for i = 20:15:80
    for j = 20:15:80
        scatter(XT(i,j), YT(i,j), 20,'black','Marker','+'); hold on
        sig_ij = sqrtm(squeeze(Sigmas_recover(i,j,:,:)))*sinusoids;
        plot(sig_ij(1,:).*5+XT(i,j), sig_ij(2,:).*5+YT(i,j),'k');
    end
end
xlim([0,1]); ylim([0,1]); axis square; hold off
xticks(0:0.2:1); yticks(0:0.2:1); xlabel('G'); ylabel('B'); 
title(sprintf('Predicted iso-threshold contours\nbased on the Wishard process'));
set(gcf,'PaperUnits','centimeters','PaperSize',[20 20]);
saveas(gcf, 'Fitted_Isothreshold_contour.pdf');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           HELPING FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [color_Lab, color_XYZ, color_LMS] = convert_rgb_lab(param,...
    background_RGB, color_RGB)
    background_Spd = param.B_monitor*background_RGB;
    background_LMS = param.T_cones*background_Spd;
    background_XYZ = param.M_LMSToXYZ*background_LMS;

    %RGB -> SPD
    color_Spd      = param.B_monitor*color_RGB;
    %SPD -> LMS
    color_LMS      = param.T_cones*color_Spd;
    %LMS -> XYZ
    color_XYZ      = param.M_LMSToXYZ*color_LMS;
    %XYZ -> Lab
    color_Lab      = XYZToLab(color_XYZ, background_XYZ);
end

function coeffs = compute_chebyshev_basis_coeffs(max_deg)
    syms x
    T = chebyshevT(0:max_deg-1, x);
    coeffs = zeros(max_deg, max_deg);
    for p = 1:max_deg
        coeffs_p = sym2poly(T(p));
        coeffs(p,(max_deg-length(coeffs_p)+1):end) = coeffs_p;
    end
end

function [U, phi] = compute_U(poly_chebyshev, W,xt,yt)
    global max_deg 
    num_grid_pts1 = size(xt,1);
    num_grid_pts2 = size(xt,2);
    nDims     = size(W,3);
    eDims   = size(W,4)-nDims;
    [val_xt, val_yt] = deal(NaN(num_grid_pts1, num_grid_pts2, max_deg));
    for d = 1:max_deg 
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
    %size of phi: num_grid_pts x num_grid_pts x max_deg x max_deg
    %size of W:   max_deg   x max_deg   x nDims   x (nDims+eDims)
    if ~isempty(W)
        U = tensorprod(phi,W,[3,4],[1,2]); %w is eta in some of the equations
        % U = NaN(num_grid_pts,num_grid_pts,nDims,nDims+eDims);
        % for i = 1:num_grid_pts
        %     for j = 1:num_grid_pts
        %         for k = 1:nDims
        %             for l = 1:(nDims+eDims)
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
    global nDims eDims NUM_MC_SAMPLES
    num_grid_pts1   = size(x,1);
    num_grid_pts2   = size(x,2);
    %compute Sigma for the reference and the comparison stimuli
    U              = compute_U(coeffs_chebyshev,W_true, x(:,:,1), x(:,:,2)); 
    Ubar           = compute_U(coeffs_chebyshev,W_true, xbar(:,:,1), xbar(:,:,2)); 
    [Sigma, Sigma_bar] = deal(NaN(num_grid_pts1, num_grid_pts2, nDims, nDims));
    for i = 1:nDims
        for j = 1:nDims
            Sigma(:,:,i,j) = sum(U(:,:,i,:).*U(:,:,j,:),4); 
            Sigma_bar(:,:,i,j) = sum(Ubar(:,:,i,:).*Ubar(:,:,j,:),4); 
        end
    end

    %simulate values for the reference and the comparison stimuli
    etas_s       = etas(1,:,:,:); 
    etas_s       = reshape(etas_s, [1, nDims+eDims,NUM_MC_SAMPLES,1]);
    z_s_noise    = tensorprod(U, squeeze(etas_s), 4, 1);
    z_s          = repmat(x,[1,1,1,NUM_MC_SAMPLES]) + z_s_noise; 
    z_s          = permute(z_s,[1,2,4,3]);

    etas_sbar    = etas(2,:,:,:); 
    etas_sbar    = reshape(etas_sbar, [1,nDims+eDims,NUM_MC_SAMPLES,1]);
    z_sbar_noise = tensorprod(Ubar, squeeze(etas_sbar), 4, 1);
    z_sbar       = repmat(xbar,[1,1,1,NUM_MC_SAMPLES]) + z_sbar_noise;
    z_sbar       = permute(z_sbar,[1,2,4,3]);
    %visualize it
    % plot_heatmap(x); plot_heatmap(z_s_noise(:,:,:,1));
    % plot_heatmap(z_s(:,:,1,:));
    % plot_heatmap(xbar); plot_heatmap(z_sbar_noise(:,:,:,1));
    % plot_heatmap(z_sbar(:,:,1,:));

    %concatenate z_s and z_sbar
    z = NaN(num_grid_pts1, num_grid_pts2, 2*NUM_MC_SAMPLES, nDims);
    z(:,:,1:NUM_MC_SAMPLES,:) = z_s;
    z(:,:,(NUM_MC_SAMPLES+1):end,:) = z_sbar;
    
    %compute prob
    pIncorrect = NaN(num_grid_pts1, num_grid_pts2);
    for t = 1:num_grid_pts1
        for v = 1:num_grid_pts2
            %predicting error probability
            % p = mvnpdf(squeeze(z(t,v,:,:)), squeeze(x(t,v,:))', ...
            %     squeeze(Sigma(t,v,:,:)));
            % q = mvnpdf(squeeze(z(t,v,:,:)), squeeze(xbar(t,v,:))', ...
            %     squeeze(Sigma_bar(t,v,:,:)));
            % pIncorrect(t,v) = mean(min(p,q)./(p+q));
            
            pC_x = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(x(t,v,:))',...
                squeeze(Sigma(t,v,:,:)));
            pInc_x = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(x(t,v,:))',...
                squeeze(Sigma(t,v,:,:)));
            pC_xbar = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(xbar(t,v,:))',...
                squeeze(Sigma_bar(t,v,:,:)));
            pInc_xbar = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(xbar(t,v,:))',...
                squeeze(Sigma_bar(t,v,:,:)));
            pC = pC_x.*pC_xbar;
            pInc = pInc_x.*pInc_xbar;
            p_normalization = pC + pInc;
            pIncorrect(t,v) = mean(pInc./p_normalization);
        end
    end
end

function nLogL = estimate_loglikelihood(w_colvec, x, xbar, y, poly_chebyshev, etas)
    global max_deg nDims eDims NUM_MC_SAMPLES
    W = reshape(w_colvec, [max_deg, max_deg, nDims,nDims+ eDims]);
    pInc = predict_error_prob(x, xbar, poly_chebyshev, W, etas);
    pC = 1 - pInc;
    logL = y.*log(pC(:) + 1e-20) + (1-y).*log(pInc(:) + 1e-20);
    nLogL = -sum(logL(:));
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
