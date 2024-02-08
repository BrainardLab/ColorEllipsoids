function pIncorrect = predict_error_prob(W, coeffs_chebyshev, x,xbar, ...
    varargin)

    p = inputParser;
    p.addParameter('etas', [], @(x)(isnumeric(x)));

    parse(p, varargin{:});
    etas    = p.Results.etas;

    n_pts1   = size(x,1);
    n_pts2   = size(x,2);
    nDims    = size(W,3);
    eDims    = size(W,4)-nDims;
    max_deg  = size(W,1);

    if ~isempty(etas)
        assert(size(etas,1)==2,'Must provide for both reference and comparison stimuli!');
        num_MC_samples = size(etas,4);
    end

    %compute Sigma for the reference and the comparison stimuli
    U              = compute_U(coeffs_chebyshev,W, x(:,:,1), x(:,:,2), max_deg); 
    Ubar           = compute_U(coeffs_chebyshev,W, xbar(:,:,1), xbar(:,:,2), max_deg); 
    [Sigma, Sigma_bar] = deal(NaN(n_pts1, n_pts2, nDims, nDims));
    for i = 1:nDims
        for j = 1:nDims
            Sigma(:,:,i,j) = sum(U(:,:,i,:).*U(:,:,j,:),4); 
            Sigma_bar(:,:,i,j) = sum(Ubar(:,:,i,:).*Ubar(:,:,j,:),4); 
        end
    end

    if ~isempty(etas) %DETERMINISTIC
        etas_s       = etas(1,:,:,:); 
        etas_s       = reshape(etas_s, [1, nDims+eDims,num_MC_samples,1]);
        etas_sbar    = etas(2,:,:,:); 
        etas_sbar    = reshape(etas_sbar, [1,nDims+eDims,...
                        num_MC_samples,1]);
    else %STOCHASTIC
        etas_s = 0.01.*randn([1,nDims+eDims, num_MC_samples,1]);
        etas_sbar = 0.01.*randn([1,nDims+eDims, num_MC_samples,1]);
    end
    %simulate values for the reference and the comparison stimuli
    z_s_noise    = tensorprod(U, squeeze(etas_s), 4, 1);
    z_s          = repmat(x,[1,1,1,num_MC_samples]) + z_s_noise; 
    z_s          = permute(z_s,[1,2,4,3]);

    z_sbar_noise = tensorprod(Ubar, squeeze(etas_sbar), 4, 1);
    z_sbar       = repmat(xbar,[1,1,1,num_MC_samples]) + z_sbar_noise;
    z_sbar       = permute(z_sbar,[1,2,4,3]);
    %visualize it
    % plot_multiHeatmap(x); plot_multiHeatmap(z_s_noise(:,:,:,1));
    % plot_multiHeatmap(z_s(:,:,1,:));
    % plot_multiHeatmap(xbar); plot_multiHeatmap(z_sbar_noise(:,:,:,1));
    % plot_multiHeatmap(z_sbar(:,:,1,:));

    %concatenate z_s and z_sbar
    z = NaN(n_pts1, n_pts2, 2*num_MC_samples, nDims);
    z(:,:,1:num_MC_samples,:) = z_s;
    z(:,:,(num_MC_samples+1):end,:) = z_sbar;
    
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