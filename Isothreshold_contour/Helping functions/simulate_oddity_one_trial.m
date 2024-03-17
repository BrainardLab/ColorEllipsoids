function diff_z1z0_to_zref = simulate_oddity_one_trial(mref, m0, m1, Uref,...
    U0, U1, num_samples, diag_term)

    %Generate random draws from isotropic, standard gaussians
    nnref = randn(num_samples, size(U1,2));
    nn0   = randn(num_samples, size(U1,2));
    nn1   = randn(num_samples, size(U1,2));

    extraN_nnref = randn(num_samples, length(m1));
    extraN_nn0   = randn(num_samples, length(m1));
    extraN_nn1   = randn(num_samples, length(m1));

    %re-scale and translate the noisy samples to have the correct mean and
    %covariance.
    zref = tensorprod(Uref', nnref, 4, 1) + mref + (extraN_nnref .* sqrt(diag_term));
    z0 = tensorprod(U0', nn0, 4, 1) + m0 + (extraN_nn0 .* sqrt(diag_term));
    z1 = tensorprod(U1', nn1, 4, 1) + m1 + (extraN_nn1 .* sqrt(diag_term));

    %compute squared distance of each probe stimulus to reference
    z0_to_zref = sum((z0 - zref).^2, 2);
    z1_to_zref = sum((z1 - zref).^2, 2);

    %return signed difference
    diff_z1z0_to_zref = z1_to_zref - z0_to_zref;
end