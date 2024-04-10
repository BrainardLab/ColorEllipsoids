function pChoosingX1 = predict_error_prob_oddity(W, coeffs_chebyshev, xref, x0, x1, ...
    varargin)

    p = inputParser;
    p.addParameter('etas', [], @(x)(isnumeric(x)));
    p.addParameter('num_MC_samples',100,@isnumeric);
    p.addParameter('scalePhi_toRGB',true,@islogical);
    p.addParameter('bandwidth',1e-4,@isnumeric);

    parse(p, varargin{:});
    etas               = p.Results.etas;
    num_MC_samples     = p.Results.num_MC_samples;
    scalePhi_toRGB     = p.Results.scalePhi_toRGB;
    bandwidth          = p.Results.bandwidth;

    nDims    = size(W,3);
    eDims    = size(W,4)-nDims;
    max_deg  = size(W,1);

    if ~isempty(etas)
        assert(size(etas,1)==3,'Must provide for three stimuli!');
        num_MC_samples = size(etas,4);
    end

    %compute Sigma for the reference and the comparison stimuli
    Uref    = compute_U(coeffs_chebyshev,W, xref, max_deg,...
            'scalePhi_toRGB', scalePhi_toRGB); 
    U0 = compute_U(coeffs_chebyshev,W, x0, max_deg,...
            'scalePhi_toRGB', scalePhi_toRGB); 
    U1 = compute_U(coeffs_chebyshev, W, x1, max_deg,...
            'scalePhi_toRGB', scalePhi_toRGB);
    [SigmaRef, Sigma0, Sigma1] = deal(NaN(size(xref,1), size(xref,2), nDims, nDims));
    for i = 1:nDims
        for j = 1:nDims
            SigmaRef(:,:,i,j) = sum(Uref(:,:,i,:).*Uref(:,:,j,:),4); 
            Sigma0(:,:,i,j) = sum(U0(:,:,i,:).*U0(:,:,j,:),4); 
            Sigma1(:,:,i,j) = sum(U1(:,:,i,:).*U1(:,:,j,:),4);
        end
    end

    if ~isempty(etas) %DETERMINISTIC
        etasRef  = etas(1,:,:,:); 
        etasRef  = reshape(etasRef, [1, nDims+eDims,num_MC_samples,1]);
        etas0    = etas(2,:,:,:); 
        etas0    = reshape(etas0, [1,nDims+eDims, num_MC_samples,1]);
        etas1    = etas(3,:,:,:); 
        etas1    = reshape(etas1, [1,nDims+eDims, num_MC_samples,1]);
    else %STOCHASTIC
        etasRef = 0.01.*randn([1,nDims+eDims, num_MC_samples,1]);
        etas0 = 0.01.*randn([1,nDims+eDims, num_MC_samples,1]);
        etas1 = 0.01.*randn([1,nDims+eDims, num_MC_samples,1]);
    end
    %simulate values for the reference and the comparison stimuli
    zref_noise    = tensorprod(Uref, squeeze(etasRef), 4, 1);
    zref          = repmat(xref,[1,1,1,num_MC_samples]) + zref_noise; 
    zref          = permute(zref,[1,2,4,3]);

    z0_noise = tensorprod(U0, squeeze(etas0), 4, 1);
    z0       = repmat(x0,[1,1,1,num_MC_samples]) + z0_noise;
    z0       = permute(z0,[1,2,4,3]);

    z1_noise = tensorprod(U1, squeeze(etas1), 4, 1);
    z1       = repmat(x1,[1,1,1,num_MC_samples]) + z1_noise;
    z1       = permute(z1,[1,2,4,3]);

    %Mahalanobis distance
    Sigma_ref0 = permute(Sigma0 + SigmaRef, [3,4,1,2]);
    Sigma_ref1 = permute(Sigma1 + SigmaRef, [3,4,1,2]);
    [z0_to_zref, z1_to_zref] = deal(NaN(size(z0,1), size(z0,2), size(z0,3)));
    for t = 1:size(z0,1)
        for v = 1:size(z0,2)
            inv_Sigma_ref0_n = inv(squeeze(Sigma_ref0(:,:,t,v)));
            inv_Sigma_ref1_n = inv(squeeze(Sigma_ref1(:,:,t,v)));
            for n = 1:size(z0,3)
                z0_tvn = squeeze(z0(t,v,n,:));
                z1_tvn = squeeze(z1(t,v,n,:)); 
                zref_tvn = squeeze(zref(t,v,n,:));
                z0_to_zref(t,v,n) = sqrt((z0_tvn - zref_tvn)' * inv_Sigma_ref0_n * (z0_tvn - zref_tvn));
                z1_to_zref(t,v,n) = sqrt((z1_tvn - zref_tvn)' * inv_Sigma_ref1_n * (z1_tvn - zref_tvn));
            end
        end
    end

    % %compute squared distance of each probe stimulus to reference
    % z0_to_zref = sum((z0 - zref).^2, 4);
    % z1_to_zref = sum((z1 - zref).^2, 4);

    %return signed difference
    diff_z1z0_to_zref = z1_to_zref - z0_to_zref;

    %evaluate one minus the cumulative density function at 0, which gives 
    %the probability of choosing the x0
    % logit_density = @(x, mu) exp(-(x-mu)./bandwidth)./ (bandwidth.* (1 + exp(-(x-mu)./bandwidth)).^2); 
    % pChoosingX0 = NaN(1, size(xref, 2));
    % for i = 1:size(xref, 2)
    %     diff_z1z0_to_zref_i = sort(squeeze(diff_z1z0_to_zref(1,i,:)),'ascend');
    %     min_diff   = diff_z1z0_to_zref_i(1);
    %     max_diff   = diff_z1z0_to_zref_i(end);
    %     range_diff = max_diff - min_diff;
    %     x_i        = linspace(min_diff - range_diff/2, max_diff + range_diff/2, 100); 
    %     conv_i     = zeros(1,100);
    %     for n = 1:size(zref, 3)
    %         log_density_n = logit_density(x_i, diff_z1z0_to_zref_i(n));
    %         log_density_n(isnan(log_density_n)) = 0;
    %         conv_i = conv_i + log_density_n;
    %     end
    %     ecdf_i = cumsum(conv_i);
    %     ecdf_i = ecdf_i./ecdf_i(end);
    % 
    %     [~,min_idx_0] = min(abs(x_i));
    %     pChoosingX0(i) = ecdf_i(min_idx_0);
    % end

    %compute prob
    pChoosingX1 = sum(diff_z1z0_to_zref>0, 3)'./size(diff_z1z0_to_zref, 3);
    %pChoosingX1 = 1 - pChoosingX0;
end