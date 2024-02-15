function [contour_outer_unscaled, contour_inner_unscaled, contour_outer,...
    contour_inner] = compute_contour_CI(M_contour, ref_xy)
    nDims     = size(M_contour,2);
    nTheta    = size(M_contour,1);
    nContours = size(M_contour,3);
    L2norm    = NaN(nContours, nTheta);
    for n = 1:nContours
        M_n = squeeze(M_contour(:,:,n));
        L2norm(n,:) = sqrt(M_n(:,1).^2 + M_n(:,2).^2);
    end
    [~, idx_lb] = min(L2norm);
    [~, idx_ub] = max(L2norm);

    [contour_outer_unscaled, contour_inner_unscaled, contour_outer, ...
        contour_inner] = deal(NaN(nTheta, nDims));
    for n = 1:nTheta
        contour_outer_unscaled(n,:) = M_contour(n,:,idx_ub(n));
        contour_inner_unscaled(n,:) = M_contour(n,:,idx_lb(n));
        contour_outer(n,:) = (M_contour(n,:,idx_ub(n)) - ref_xy).*5 + ref_xy;
        contour_inner(n,:) = (M_contour(n,:,idx_lb(n)) - ref_xy).*5 + ref_xy;
    end
end