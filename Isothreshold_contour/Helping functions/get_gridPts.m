function grid_pts = get_gridPts(X, Y, fixed_dim, val_fixed_dim)
    XY = {X,Y};
    grid_pts = cell(1,length(fixed_dim));
    for i = 1:length(fixed_dim)
        varying_dim = setdiff(1:3, fixed_dim(i));
        grid_pts_i = cell(1,3);
        grid_pts_i{fixed_dim(i)} = val_fixed_dim(i).*ones(size(X));
        for j = 1:length(varying_dim)
            grid_pts_i{varying_dim(j)} = XY{j};
        end
        grid_pts{i} = cat(3, grid_pts_i{:});
    end
end
