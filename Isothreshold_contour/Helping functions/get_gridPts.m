function grid_pts = get_gridPts(X, Y, fixed_dim, val_fixed_dim)
%GET_GRIDPTS Generates grid points for RGB values with one dimension fixed.
%
% This function returns a cell array of grid points for RGB combinations
% when one of the R, G, or B dimensions is fixed to a specific value. The
% grid points are generated based on the input ranges for the two varying
% dimensions.
%
% Parameters:
% X - A vector of values for the first varying dimension.
% Y - A vector of values for the second varying dimension.
% fixed_dim - An array of indices (1 for R, 2 for G, 3 for B) indicating 
%             which dimension(s) to fix. Can handle multiple dimensions.
% val_fixed_dim - An array of values corresponding to each fixed dimension
%                 specified in `fixed_dim`. Each value in `val_fixed_dim`
%                 is used to fix the value of the corresponding dimension.
%
% Returns:
% grid_pts - A cell array where each cell contains a 3D matrix of grid 
%            points. Each matrix corresponds to a set of RGB values where 
%            one dimension is fixed. The size of each matrix is determined 
%            by the lengths of X and Y, with the third dimension representing
%            the RGB channels.

    XY = {X,Y};

    % Initialize a cell array to hold the grid points for each fixed dimension.
    grid_pts = cell(1,length(fixed_dim));

    % Loop through each fixed dimension specified.
    for i = 1:length(fixed_dim)
        % Determine the dimensions that will vary.
        varying_dim = setdiff(1:3, fixed_dim(i));
        % Initialize a cell array to hold the current set of grid points.
        grid_pts_i = cell(1,length(fixed_dim));
        % Set the fixed dimension to its specified value across all grid points.
        grid_pts_i{fixed_dim(i)} = val_fixed_dim(i).*ones(size(X));

        % Assign the input ranges to the varying dimensions.
        for j = 1:length(varying_dim)
            grid_pts_i{varying_dim(j)} = XY{j};
        end

        % Concatenate the individual dimension arrays into a 3D matrix and
        % store it in the output cell array.
        grid_pts{i} = cat(length(fixed_dim), grid_pts_i{:});
    end
end
