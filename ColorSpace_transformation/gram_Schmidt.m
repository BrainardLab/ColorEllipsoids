function [e1, e2] = gram_Schmidt(v1, v2)
% Create nSamples unit vectors uniformly spaced in angle within the plane
% spanned by v1 and v2 (Gram–Schmidt orthonormalization).
    v1 = v1(:);
    v2 = v2(:);
    e1 = v1 / norm(v1);
    v2p = v2 - (e1' * v2) * e1;
    nv2 = norm(v2p);
    if nv2 < 1e-14
        error('v1 and v2 are nearly collinear; cannot span a plane.');
    end
    e2 = v2p / nv2;
end