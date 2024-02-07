
extrap_nGridPts_ref = linspace(0.275, 0.725,4);
[g,b] = meshgrid(extrap_nGridPts_ref, extrap_nGridPts_ref);
r = ones(size(g)).*0.5;
extrap_ref_points = NaN(4,4,3);
extrap_ref_points(:,:,1) = r;
extrap_ref_points(:,:,2) = g; 
extrap_ref_points(:,:,3) = b;

%initialize
[extrap_recover_rgb_comp_est,extrap_recover_rgb_contour] = deal(NaN(4,4,stim.numDirPts-1,2));
extrap_recover_rgb_contour_cov = NaN(4,4,2,2);
[extrap_recover_fitEllipse, extrap_recover_fitEllipse_unscaled] = deal(NaN(4,4,200,2));

%for each reference stimulus
for i = 1:4
    for j = 1:4
        %grab the reference stimulus's RGB
        rgb_ref_ij = extrap_ref_points(i,j,:);
        rgb_ref_ij_t = extrap_ref_points(i,j,sim.varying_RGBplane);
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
            extrap_recover_rgb_comp_est(i,j,k,:) = rgb_ref_ij_s + ...
                results.contour_scaler.*vecDir.*fits.vecLength(min_idx);
        end

        % compute the iso-threshold contour 
        rgb_contour_lpij = squeeze(extrap_recover_rgb_comp_est(i,j,:,:));
        extrap_recover_rgb_contour_cov(i,j,:,:) = cov(rgb_contour_lpij);

        %fit an ellipse
        [~,fitA_lpij,~,fitQ_lpij] = FitEllipseQ(rgb_contour_lpij' - ...
            rgb_ref_ij_s,'lockAngleAt0',false);
        extrap_recover_fitEllipse(i,j,:,:) = (PointsOnEllipseQ(...
            fitQ_lpij,plt.circleIn2D) + rgb_ref_ij_s)';

        %un-scaled
        extrap_recover_fitEllipse_unscaled(i,j,:,:) = (squeeze(extrap_recover_fitEllipse(i,j,:,:)) -...
            rgb_ref_ij_s')./results.contour_scaler + rgb_ref_ij_s';
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
