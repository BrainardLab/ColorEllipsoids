function coeffs = compute_chebyshev_basis_coeffs(max_deg)
    assert(max_deg >= 1, ['The maximum degree of chebyshev polynomial has ',...
        'to be equal or greater than 1! ']);

    syms x
    T = chebyshevT(0:max_deg-1, x);
    coeffs = zeros(max_deg, max_deg);
    for p = 1:max_deg
        coeffs_p = sym2poly(T(p));
        coeffs(p,(max_deg-length(coeffs_p)+1):end) = coeffs_p;
    end
end