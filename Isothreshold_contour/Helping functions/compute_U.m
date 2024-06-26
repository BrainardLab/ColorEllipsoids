function [U, phi] = compute_U(poly_chebyshev, W, M, max_deg, varargin)
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

    maxDeg = size(poly_chebyshev,1);
    nDims  = size(W,3);
    eDims  = size(W,4)-nDims;

    size_M = size(M);
    if size_M(end) == 2
        n_pts1 = size_M(1);
        n_pts2 = size_M(2);

        % Evaluate Chebyshev polynomials for each dimension
        [val_xt, val_yt] = deal(NaN(n_pts1, n_pts2, max_deg));
        for d = 1:max_deg 
            val_xt(:,:,d) = polyval(poly_chebyshev(d,:),M(:,:,1));
            val_yt(:,:,d) = polyval(poly_chebyshev(d,:),M(:,:,2));
        end
        % Replicate and permute values for element-wise multiplication
        val_xt_repmat = repmat(val_xt, [1,1,1,maxDeg]);
        val_yt_repmat = permute(repmat(val_yt, [1,1,1,maxDeg]),[1,2,4,3]);
        % Compute the basis function (phi) by multiplying the Chebyshev values
        phi_raw = val_xt_repmat.*val_yt_repmat;

    elseif size_M(end) == 3
        n_pts1 = size_M(1);
        n_pts2 = size_M(2);
        n_pts3 = size_M(3);

        % Evaluate Chebyshev polynomials for each dimension
        [val_xt, val_yt, val_zt] = deal(NaN(n_pts1, n_pts2, n_pts3, max_deg));
        for d = 1:max_deg 
            val_xt(:,:,:,d) = polyval(poly_chebyshev(d,:),M(:,:,:,1));
            val_yt(:,:,:,d) = polyval(poly_chebyshev(d,:),M(:,:,:,2));
            val_zt(:,:,:,d) = polyval(poly_chebyshev(d,:),M(:,:,:,3));
        end
        % Replicate and permute values for element-wise multiplication
        % Extend val_xt across two additional dimensions to prepare for 3D multiplication. 
        % This replication creates a 6D array where the 5th and 6th dimensions are copies 
        % of val_xt to match each degree combination in x, y, and z.
        val_xt_repmat = repmat(val_xt, [1,1,1,1,maxDeg,maxDeg]);

        % Extend val_yt similar to val_xt but permute the dimensions to align the y degrees 
        % across the 5th dimension (previously the 4th dimension in val_xt_repmat). 
        % This prepares val_yt for correct element-wise multiplication with val_xt and 
        % val_zt, ensuring that each combination of degrees in x, y is represented.
        
        val_yt_repmat = permute(repmat(val_yt, [1,1,1,1,maxDeg,maxDeg]),[1,2,3,5,4,6]);
        % Extend val_zt in the same way as val_yt, but align the z degrees across the 6th dimension, 
        % which is the unique dimension for val_zt to ensure that multiplication covers all 
        % combinations of degrees in x, y, and z.
        % This alignment is crucial for the subsequent element-wise multiplication to correctly 
        % compute the 3D basis functions.
        val_zt_repmat = permute(repmat(val_zt, [1,1,1,1,maxDeg,maxDeg]),[1,2,3,6,5,4]);
        % Compute the basis function (phi) by multiplying the Chebyshev values
        phi_raw = val_xt_repmat.*val_yt_repmat.*val_zt_repmat;
    end
    %sometimes we want to rescale phi so that its range is not -1 to 1, but
    %0 to 1 for example
    if scalePhi_toRGB
        phi = (phi_raw + 1)./2; %rescale it: [-1 1] in chebyshev space to [0 1] in RGB space
    else
        phi = phi_raw;
    end

    %visualize it
    if visualize_xt && size_M(end) == 2; plot_multiHeatmap(val_xt_repmat,'permute_M',true); end
    if visualize_yt && size_M(end) == 2; plot_multiHeatmap(val_yt_repmat,'permute_M',true); end
    if visualize_phi && size_M(end) == 2; plot_multiHeatmap(phi,'permute_M',true); end
    if visualize_W && size_M(end) == 2; plot_multiHeatmap(W,'permute_M',true); end

    %equivalent of np.einsum(ij,jk-ik',A,B)
    %size of phi: num_grid_pts x num_grid_pts x max_deg x max_deg
    %size of W:   max_deg   x max_deg   x nDims   x (nDims+eDims)
    if ~isempty(W)
        if size_M(end) == 2
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
        elseif size_M(end) == 3
            U = tensorprod(phi,W,[4,5,6],[1,2,3]);
        end
    else
        U = [];
    end
end