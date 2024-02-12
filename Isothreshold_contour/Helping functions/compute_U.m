function [U, phi] = compute_U(poly_chebyshev, W,xt,yt, max_deg, varargin)
    p = inputParser;
    p.addParameter('visualize_xt', false, @islogical);
    p.addParameter('visualize_yt', false, @islogical);
    p.addParameter('visualize_phi', false, @islogical);
    p.addParameter('visualize_W', false, @islogical);
    p.addParameter('scalePhi_toRGB',true, @islogical);

    parse(p, varargin{:});
    visualize_xt         = p.Results.visualize_xt;
    visualize_yt         = p.Results.visualize_yt;
    visualize_phi        = p.Results.visualize_phi;
    visualize_W          = p.Results.visualize_W;
    scalePhi_toRGB       = p.Results.scalePhi_toRGB;

    n_pts1 = size(xt,1);
    n_pts2 = size(xt,2);
    nDims  = size(W,3);
    eDims  = size(W,4)-nDims;

    [val_xt, val_yt] = deal(NaN(n_pts1, n_pts2, max_deg));
    for d = 1:max_deg 
        val_xt(:,:,d) = polyval(poly_chebyshev(d,:),xt);
        val_yt(:,:,d) = polyval(poly_chebyshev(d,:),yt);
    end

    val_xt_repmat = repmat(val_xt, [1,1,1,size(poly_chebyshev,2)]);
    val_yt_repmat = permute(repmat(val_yt, [1,1,1,size(poly_chebyshev,2)]),[1,2,4,3]);
    phi_raw = val_xt_repmat.*val_yt_repmat;
    %sometimes we want to rescale phi so that its range is not -1 to 1, but
    %0 to 1 for example
    if scalePhi_toRGB
        phi = (phi_raw + 1)./2; %rescale it: [-1 1] in chebyshev space to [0 1] in RGB space
    else
        phi = phi_raw;
    end

    %visualize it
    if visualize_xt; plot_multiHeatmap(val_xt_repmat,'permute_M',true); end
    if visualize_yt; plot_multiHeatmap(val_yt_repmat,'permute_M',true); end
    if visualize_phi; plot_multiHeatmap(phi,'permute_M',true); end
    if visualize_W; plot_multiHeatmap(W,'permute_M',true); end

    %equivalent of np.einsum(ij,jk-ik',A,B)
    %size of phi: num_grid_pts x num_grid_pts x max_deg x max_deg
    %size of W:   max_deg   x max_deg   x nDims   x (nDims+eDims)
    if ~isempty(W)
        U = tensorprod(phi,W,[3,4],[1,2]); %w is eta in some of the equations
        % U = NaN(n_pts1,n_pts2,nDims,nDims+eDims);
        % for i = 1:n_pts1
        %     for j = 1:n_pts2
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