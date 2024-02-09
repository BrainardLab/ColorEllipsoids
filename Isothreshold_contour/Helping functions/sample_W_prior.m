function W = sample_W_prior(max_deg, num_dims, extra_dims, var_scale, decay_rate)

    degs = repmat(0:(max_deg-1),[max_deg,1]) + repmat((0:(max_deg-1))',[1,max_deg]);
    vars = var_scale.*(decay_rate.^degs);
    stds = sqrt(vars);
    W = repmat(stds,[1,1,num_dims, num_dims+extra_dims]).*...
        randn([max_deg, max_deg, num_dims, (num_dims+extra_dims)]);
end