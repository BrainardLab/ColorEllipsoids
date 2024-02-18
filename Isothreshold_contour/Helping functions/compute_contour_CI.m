function [contour_outer_unscaled, contour_inner_unscaled, contour_outer,...
    contour_inner] = compute_contour_CI(M_contour, ref_xy)
    %M_contour has to have a size of nTheta x nDims x nContours
    nTheta    = size(M_contour,1);
    nDims     = size(M_contour,2);
    nContours = size(M_contour,3);
    L2norm    = NaN(nTheta,nContours);
    [idx_lb, idx_ub] = deal(NaN(1,nTheta));
    for n = 1:nTheta
        M_n = squeeze(M_contour(n,:,:)) - ref_xy';
        L2norm(n,:) = sqrt(M_n(1,:).^2 + M_n(2,:).^2);
        [~,idx_lb(n)] = min(L2norm(n,:)); 
        [~,idx_ub(n)] = max(L2norm(n,:)); 
    end

    [contour_outer_unscaled, contour_inner_unscaled, contour_outer, ...
        contour_inner] = deal(NaN(nTheta, nDims));
    % figure;
    for n = 1:nTheta
        contour_outer_unscaled(n,:) = M_contour(n,:,idx_ub(n));
        contour_inner_unscaled(n,:) = M_contour(n,:,idx_lb(n));
        contour_outer(n,:) = (M_contour(n,:,idx_ub(n)) - ref_xy).*5 + ref_xy;
        contour_inner(n,:) = (M_contour(n,:,idx_lb(n)) - ref_xy).*5 + ref_xy;
        % plot(contour_outer_unscaled(:,1), contour_outer_unscaled(:,2)); hold on
    end

end