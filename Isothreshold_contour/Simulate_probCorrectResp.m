clear all; close all; clc

%% load isothreshold contours simulated based on CIELAB
load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% First create a cube and select the RG, the RB and the GB planes
sim.slc_fixedVal     = 0.5; %the level of the fixed plane
sim.slc_fixedVal_idx = find(stim.fixed_RGBvec == sim.slc_fixedVal);
sim.slc_RGBplane     = 1; %GB plane with a fixed R value
sim.varying_RGBplane = setdiff(1:3, sim.slc_RGBplane);
sim.plane_points     = param.plane_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.ref_points       = stim.ref_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.background_RGB   = stim.background_RGB(:,sim.slc_fixedVal_idx);
sim.fitA             = squeeze(results.fitA(sim.slc_fixedVal_idx,sim.slc_RGBplane,:,:,:,:));

%% Simulate data given the iso-threshold contours
sim.alpha               = 1.1729;
sim.beta                = 1.2286;
sim.pC_given_alpha_beta = ComputeWeibTAFC(stim.deltaE_1JND,sim.alpha,sim.beta);%0.8;
sim.nSims_perDir        = 50; %5 x 16 = 80; 15 x 16 = 240; 50 x 16 = 800
sim.nSims               = sim.nSims_perDir * (stim.numDirPts -1);
sim.perturb_factor      = 1; %1: no perturbation
sim.method_sampling     = 'Random';

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_pij = squeeze(sim.ref_points(i,j,:));
        %convert it to Lab
        [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param, sim.background_RGB, rgb_ref_pij);
        sim.ref_Lab(i,j,:) = ref_Lab_lpij;
        
        %simulate comparison stimulus
        if strcmp(sim.method_sampling, 'NearContour')
            sim.rgb_comp(i,j,:,:) = draw_rgb_comp(stim, sim, ...
                squeeze(results.opt_vecLen(sim.slc_fixedVal_idx,...
                sim.slc_RGBplane, i, j, :, :)),...
                squeeze(results.rgb_contour_cov(sim.slc_fixedVal_idx,...
                sim.slc_RGBplane, i, j, :, :,:)),...
                rgb_ref_pij(sim.varying_RGBplane));
        elseif strcmp(sim.method_sampling, 'Random')
            sim.rgb_comp(i,j,:,:) = draw_rgb_comp_random(sim, ...
                rgb_ref_pij(sim.varying_RGBplane), [-0.025, 0.025]);
        end
        
        %simulate binary responses
        for n = 1:sim.nSims
            [sim.lab_comp(i,j,:,n), ~, ~] = convert_rgb_lab(param, ...
                sim.background_RGB, squeeze(sim.rgb_comp(i,j,:,n)));
            sim.deltaE(i,j,n) = norm(squeeze(sim.lab_comp(i,j,:,n)) - ref_Lab_lpij);
            sim.probC(i,j,n) = ComputeWeibTAFC(sim.deltaE(i,j,n), ...
                sim.alpha, sim.beta);
        end
        sim.resp_binary(i,j,:) = binornd(1, squeeze(sim.probC(i,j,:)), [sim.nSims, 1]);
    end
end

%% define model specificity for the wishart process
model.max_deg        = 3;     %corresponds to p in Alex's document
model.nDims          = 2;     %corresponds to a = 1, a = 2 
model.eDims          = 0;     %extra dimensions
model.num_grid_pts   = 100;
model.num_MC_samples = 100;   %number of monte carlo simulation

model.coeffs_chebyshev = compute_chebyshev_basis_coeffs(model.max_deg);
disp(model.coeffs_chebyshev)

model.xt = linspace(-1,1,model.num_MC_samples);
model.yt = linspace(-1,1,model.num_MC_samples);
[model.XT, model.YT]   = meshgrid(model.xt, model.yt);
[~, model.M_chebyshev] = compute_U(model.coeffs_chebyshev,[],...
    model.XT,model.YT, model.max_deg);
%visualize it
plot_heatmap(model.M_chebyshev)

%% Fit the model to the data
%comparison stimulus
x_sim            = sim.rgb_comp(:,:,sim.varying_RGBplane,:);
x_sim_d1         = squeeze(x_sim(:,:,1,:));
x_sim_d2         = squeeze(x_sim(:,:,2,:));
sim.x_sim_org(1,:,1) = x_sim_d1(:); 
sim.x_sim_org(1,:,2) = x_sim_d2(:);

%reference stimulus
xbar_sim         = repmat(sim.ref_points(:,:,sim.varying_RGBplane),...
                    [1,1,1,sim.nSims]);
xbar_sim_d1      = squeeze(xbar_sim(:,:,1,:));
xbar_sim_d2      = squeeze(xbar_sim(:,:,2,:));
sim.xbar_sim_org(1,:,1) = xbar_sim_d1(:); 
sim.xbar_sim_org(1,:,2) = xbar_sim_d2(:);
% if we just want to use one set of randomly drawn samples
% sim.etas = 0.01.*randn([2,1,model.nDims + model.eDims, model.num_MC_samples]);

%define objective functions
objectiveFunc = @(w_colvec) estimate_loglikelihood(w_colvec,sim.xbar_sim_org,...
    sim.x_sim_org, model, sim);

%% call bads 
fits.nFreeParams = model.max_deg*model.max_deg*model.nDims*(model.nDims + model.eDims);
fits.lb      = -0.05.*ones(1, fits.nFreeParams);
fits.ub      =  0.05.*ones(1, fits.nFreeParams);
fits.plb     = -0.01.*ones(1, fits.nFreeParams);
fits.pub     =  0.01.*ones(1, fits.nFreeParams);
%have different initial points to avoid fmincon from getting stuck at
%some places
fits.N_runs  = 1;
fits.init    = rand(fits.N_runs,fits.nFreeParams).*(fits.pub-fits.plb) + fits.plb;

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
[fits.U_recover,~] = compute_U(model.coeffs_chebyshev, fits.w_est_best, ...
    param.x_grid, param.y_grid, model.max_deg); 
for i = 1:model.nDims
    for j = 1:model.nDims
        fits.Sigmas_recover(:,:,i,j) = sum(fits.U_recover(:,:,i,:).*...
            fits.U_recover(:,:,j,:),4);
    end
end

%% make predictions on the probability of correct
fits.ngrid_bruteforce = 1e3;
fits.vecLength = linspace(0,max(results.opt_vecLen(:))*2,fits.ngrid_bruteforce);

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_ij = sim.ref_points(i,j,:);
        rgb_ref_ij_t = sim.ref_points(i,j,sim.varying_RGBplane);
        rgb_ref_ij_s = squeeze(rgb_ref_ij_t);
        
        %for each chromatic direction
        for k = 1:stim.numDirPts-1
            %determine the direction we are going 
            vecDir = stim.grid_theta_xy(:,k);
            
            %run fmincon to search for the magnitude of vector that
            %leads to a pre-determined deltaE
            for l = 1:fits.ngrid_bruteforce
                rgb_comp_pijk = rgb_ref_ij_t;
                rgb_comp_pijk(1,1,:) = squeeze(rgb_comp_pijk(1,1,:)) + ...
                    vecDir.*fits.vecLength(l);
                pInc(l) = predict_error_prob(fits.w_est_best,...
                    rgb_ref_ij_t,rgb_comp_pijk, model, sim);
            end
            [~, min_idx] = min(abs((1-pInc) - sim.pC_given_alpha_beta));
            fits.recover_rgb_comp_est(i,j,k,:) = rgb_ref_ij_s + ...
                results.contour_scaler.*vecDir.*fits.vecLength(min_idx);
        end

        % compute the iso-threshold contour 
        rgb_contour_lpij = squeeze(fits.recover_rgb_comp_est(i,j,:,:));
        fits.recover_rgb_contour(i,j,:,:) = rgb_contour_lpij';
        fits.recover_rgb_contour_cov(i,j,:,:) = cov(rgb_contour_lpij);

        %fit an ellipse
        [~,fitA_lpij,~,fitQ_lpij] = FitEllipseQ(rgb_contour_lpij' - ...
            rgb_ref_ij_s,'lockAngleAt0',false);
        fits.recover_fitA(i,j,:,:) = fitA_lpij;
        fits.recover_fitQ(i,j,:,:) = fitQ_lpij;
        fits.recover_fitEllipse(i,j,:,:) = (PointsOnEllipseQ(...
            fitQ_lpij,plt.circleIn2D) + rgb_ref_ij_s)';

        %un-scaled
        fits.recover_fitEllipse_unscaled(i,j,:,:) = (squeeze(fits.recover_fitEllipse(i,j,:,:)) -...
            rgb_ref_ij_s')./results.contour_scaler + rgb_ref_ij_s';
    end
end

%% visualize
thetas = linspace(0,2*pi, plt.nThetaEllipse);
sinusoids = [cos(thetas); sin(thetas)];
figure
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        scatter(stim.grid_ref(i),stim.grid_ref(j), 20,'black','Marker','+'); hold on
        plot(squeeze(fits.recover_fitEllipse(i,j,:,1)),...
             squeeze(fits.recover_fitEllipse(i,j,:,2)),...
             'k-','lineWidth',1.5); hold on
        % sig_ij = sqrtm(squeeze(fits.Sigmas_recover(i,j,:,:)))*plt.circleIn2D;
        % plot(sig_ij(1,:).*5+param.x_grid(i,j), sig_ij(2,:).*5+param.y_grid(i,j),'k');
    end
end
xlim([0,1]); ylim([0,1]); axis square; hold off
xticks(0:0.2:1); yticks(0:0.2:1); 
if sim.slc_RGBplane == 1; xlabel('G'); ylabel('B'); 
elseif sim.slc_RGBplane == 2; xlabel('R'); ylabel('B');
else; xlabel('R'); ylabel('G');
end
title(sprintf(['Predicted iso-threshold contours \nin the ', plt.ttl{sim.slc_RGBplane},...
    ' based on the Wishart process']));
set(gcf,'PaperUnits','centimeters','PaperSize',[20 20]);
% saveas(gcf, ['Fitted_Isothreshold_contour_',plt.ttl{sim.slc_RGBplane},...
%     '_sim',num2str(sim.nSims), 'perCond.pdf']);

%% visualize the fits on top of the data
figure; cmap = colormap("gray"); colormap(flipud(cmap))
t = tiledlayout(stim.nGridPts_ref,stim.nGridPts_ref,'TileSpacing','none');
for i = stim.nGridPts_ref:-1:1  %row: B
    for j = 1:stim.nGridPts_ref %column: G
        x_axis = linspace(-0.025,0.025,50)+stim.grid_ref(j);
        y_axis = linspace(-0.025,0.025,50)+stim.grid_ref(i);

        nexttile
        %simulated trials
        if strcmp(sim.method_sampling,'NearContour')
            h = histcounts2(squeeze(sim.rgb_comp(i,j,sim.varying_RGBplane(2),:)),...
                squeeze(sim.rgb_comp(i,j, sim.varying_RGBplane(1),:)),...
                y_axis,x_axis);
            imagesc(x_axis, y_axis, h); axis square; hold on;
        elseif strcmp(sim.method_sampling, 'Random')
            idx_1 = find(sim.resp_binary(i,j,:)==1);
            idx_0 = find(sim.resp_binary(i,j,:)==0);
            scatter(squeeze(sim.rgb_comp(i,j,sim.varying_RGBplane(1),idx_1)),...
                squeeze(sim.rgb_comp(i,j, sim.varying_RGBplane(2),idx_1)),'.',...
                'MarkerFaceColor',[173,216,230]./255,'MarkerEdgeColor',[173,216,230]./255); hold on
            scatter(squeeze(sim.rgb_comp(i,j,sim.varying_RGBplane(1),idx_0)),...
                squeeze(sim.rgb_comp(i,j, sim.varying_RGBplane(2),idx_0)),'*',...
                'MarkerFaceColor',[255,179,138]./255,'MarkerEdgeColor',[255,179,138]./255);
        end

        % fits
        plot(squeeze(fits.recover_fitEllipse_unscaled(i,j,:,1)),...
             squeeze(fits.recover_fitEllipse_unscaled(i,j,:,2)),...
             '-','lineWidth',1,'Color',[76,153,0]./255); 
        % ground truth
        fitEllipse_unscaled = (squeeze(results.fitEllipse(...
            sim.slc_fixedVal_idx,sim.slc_RGBplane, i,j,:,:)) -...
            [stim.grid_ref(j), stim.grid_ref(i)])./results.contour_scaler +...
            [stim.grid_ref(j), stim.grid_ref(i)];
        plot(squeeze(fitEllipse_unscaled(:,1)),...
             squeeze(fitEllipse_unscaled(:,2)),...
             'r--','lineWidth',1);         
        hold off; box on;
        xlim([x_axis(1), x_axis(end)]); ylim([y_axis(1), y_axis(end)]);

        if j == 1; yticks(stim.grid_ref(i));
        else; yticks([]);end

        if i == 1; xticks(stim.grid_ref(j));
        else; xticks([]);end
        set(gca,'FontSize',12)
    end
end
set(gcf,'Units','Normalized','Position',[0, 0.1,0.415,0.7]);
set(gcf,'PaperUnits','centimeters','PaperSize',[35 35]);
saveas(gcf, ['Fitted_Isothreshold_contour_w',sim.method_sampling,...
    'Samples_',plt.ttl{sim.slc_RGBplane},...
    '_sim',num2str(sim.nSims), 'perCond.pdf']);

%% save the data
E = {param, stim, results, fits};
save(['Fits_isothreshold_',plt.ttl{sim.slc_RGBplane},'_sim',...
    num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,'.mat'],'E');

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

function [U, phi] = compute_U(poly_chebyshev, W,xt,yt, max_deg)
    n_pts1 = size(xt,1);
    n_pts2 = size(xt,2);
    nDims     = size(W,3);
    eDims   = size(W,4)-nDims;
    [val_xt, val_yt] = deal(NaN(n_pts1, n_pts2, max_deg));
    for d = 1:max_deg 
        val_xt(:,:,d) = polyval(poly_chebyshev(d,:),xt);
        val_yt(:,:,d) = polyval(poly_chebyshev(d,:),yt);
    end

    val_xt_repmat = repmat(val_xt, [1,1,1,size(poly_chebyshev,2)]);
    val_yt_repmat = permute(repmat(val_yt, [1,1,1,size(poly_chebyshev,2)]),[1,2,4,3]);
    phi_raw = val_xt_repmat.*val_yt_repmat;
    phi = (phi_raw + 1)./2; %rescale it: [-1 1] in chebyshev space to [0 1] in RGB space

    %equivalent of np.einsum(ij,jk-ik',A,B)
    %size of phi: num_grid_pts x num_grid_pts x max_deg x max_deg
    %size of W:   max_deg   x max_deg   x nDims   x (nDims+eDims)
    if ~isempty(W)
        U = tensorprod(phi,W,[3,4],[1,2]); %w is eta in some of the equations
    else
        U = [];
    end
end

function pIncorrect = predict_error_prob(W_true, xbar, x, model, sim)
    n_pts1   = size(x,1);
    n_pts2   = size(x,2);
    %compute Sigma for the reference and the comparison stimuli
    U              = compute_U(model.coeffs_chebyshev,W_true, ...
                        x(:,:,1), x(:,:,2), model.max_deg); 
    Ubar           = compute_U(model.coeffs_chebyshev,W_true, ...
                        xbar(:,:,1), xbar(:,:,2), model.max_deg); 
    [Sigma, Sigma_bar] = deal(NaN(n_pts1, n_pts2, ...
        model.nDims, model.nDims));
    for i = 1:model.nDims
        for j = 1:model.nDims
            Sigma(:,:,i,j) = sum(U(:,:,i,:).*U(:,:,j,:),4); 
            Sigma_bar(:,:,i,j) = sum(Ubar(:,:,i,:).*Ubar(:,:,j,:),4); 
        end
    end

    if isfield(sim,'etas') %DETERMINISTIC
        etas_s       = sim.etas(1,:,:,:); 
        etas_s       = reshape(etas_s, [1, model.nDims+model.eDims,...
                        model.num_MC_samples,1]);
        etas_sbar    = sim.etas(2,:,:,:); 
        etas_sbar    = reshape(etas_sbar, [1,model.nDims+model.eDims,...
                        model.num_MC_samples,1]);
    else %STOCHASTIC
        etas_s = 0.01.*randn([1,model.nDims+model.eDims, model.num_MC_samples,1]);
        etas_sbar = 0.01.*randn([1,model.nDims+model.eDims, model.num_MC_samples,1]);
    end
    %simulate values for the reference and the comparison stimuli
    z_s_noise    = tensorprod(U, squeeze(etas_s), 4, 1);
    z_s          = repmat(x,[1,1,1,model.num_MC_samples]) + z_s_noise; 
    z_s          = permute(z_s,[1,2,4,3]);

    z_sbar_noise = tensorprod(Ubar, squeeze(etas_sbar), 4, 1);
    z_sbar       = repmat(xbar,[1,1,1,model.num_MC_samples]) + z_sbar_noise;
    z_sbar       = permute(z_sbar,[1,2,4,3]);
    %visualize it
    % plot_heatmap(x); plot_heatmap(z_s_noise(:,:,:,1));
    % plot_heatmap(z_s(:,:,1,:));
    % plot_heatmap(xbar); plot_heatmap(z_sbar_noise(:,:,:,1));
    % plot_heatmap(z_sbar(:,:,1,:));

    %concatenate z_s and z_sbar
    z = NaN(n_pts1, n_pts2, 2*model.num_MC_samples, model.nDims);
    z(:,:,1:model.num_MC_samples,:) = z_s;
    z(:,:,(model.num_MC_samples+1):end,:) = z_sbar;
    
    %compute prob
    pIncorrect = NaN(n_pts1, n_pts2);
    for t = 1:n_pts1
        for v = 1:n_pts2
            %predicting error probability
            p = mvnpdf(squeeze(z(t,v,:,:)), squeeze(x(t,v,:))', ...
                squeeze(Sigma(t,v,:,:)));
            q = mvnpdf(squeeze(z(t,v,:,:)), squeeze(xbar(t,v,:))', ...
                squeeze(Sigma_bar(t,v,:,:)));
            pIncorrect(t,v) = mean(min(p,q)./(p+q));
            
            % pC_x = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(sim.x_sim_org(t,v,:))',...
            %     squeeze(Sigma(t,v,:,:)));
            % pInc_x = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(sim.x_sim_org(t,v,:))',...
            %     squeeze(Sigma(t,v,:,:)));
            % pC_xbar = mvnpdf(squeeze(z_sbar(t,v,:,:)), squeeze(sim.xbar_sim_org(t,v,:))',...
            %     squeeze(Sigma_bar(t,v,:,:)));
            % pInc_xbar = mvnpdf(squeeze(z_s(t,v,:,:)), squeeze(sim.xbar_sim_org(t,v,:))',...
            %     squeeze(Sigma_bar(t,v,:,:)));
            % pC = pC_x.*pC_xbar;
            % pInc = pInc_x.*pInc_xbar;
            % p_normalization = pC + pInc;
            % pIncorrect(t,v) = mean(pInc./p_normalization);
        end
    end
end

function nLogL = estimate_loglikelihood(w_colvec, xbar, x, model, sim)
    W = reshape(w_colvec, [model.max_deg, model.max_deg, model.nDims,...
        model.nDims + model.eDims]);
    pInc    = predict_error_prob(W, xbar, x, model, sim);
    pC      = 1 - pInc;
    logL    = sim.resp_binary(:).*log(pC(:) + 1e-20) + ...
                (1-sim.resp_binary(:)).*log(pInc(:) + 1e-20);
    nLogL   = -sum(logL(:));
end

function rgb_comp_sim = draw_rgb_comp(stim, sim, vecLength, cov, ref)
    rgb_comp = NaN(2,stim.numDirPts-1, sim.nSims_perDir);
    for d = 1:(stim.numDirPts-1)
        cov_proj = cov * stim.grid_theta_xy(:,d);
        noise = randn(2,sim.nSims_perDir).*sim.perturb_factor.*cov_proj;
        rgb_comp(:,d,:) = ref + stim.grid_theta_xy(:,d).*vecLength(d) + noise;
    end
    rgb_comp_sim = NaN(3, sim.nSims);
    rgb_comp_sim(sim.slc_RGBplane,:) = sim.slc_fixedVal;
    rgb_comp1 = rgb_comp(1,:,:); rgb_comp2 = rgb_comp(2,:,:);
    rgb_comp_sim(sim.varying_RGBplane(1),:) = rgb_comp1(:);
    rgb_comp_sim(sim.varying_RGBplane(2),:) = rgb_comp2(:);
end

function rgb_comp_sim = draw_rgb_comp_random(sim, ref, range)
    rgb_comp_sim                     = rand(3, sim.nSims).*(range(2)-range(1))+range(1);
    rgb_comp_sim(sim.slc_RGBplane,:) = sim.slc_fixedVal;
    rgb_comp_sim(sim.varying_RGBplane,:) = rgb_comp_sim(sim.varying_RGBplane,:) + ref;
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
