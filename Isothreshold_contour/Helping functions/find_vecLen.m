function opt_vecLen = find_vecLen(background_RGB, ref_RGB, ref_Lab, ...
    vecDir, param, stim)
% Finds the optimal vector length along a specified chromatic direction
% such that the perceptual difference (deltaE) between a reference stimulus
% and a comparison stimulus is equal to a specific value (e.g., 1 JND) in
% the CIELab color space.
%
% Inputs:
%   background_RGB - The RGB values of the background.
%   ref_RGB        - The RGB values of the reference stimulus.
%   ref_Lab        - The CIELab values of the reference stimulus.
%   vecDir         - The chromatic direction vector for comparison stimulus variation.
%   param          - A structure containing parameters necessary for the conversion
%                    and computation processes.
%   stim           - A structure containing the target deltaE value (deltaE_1JND).
%
% Output:
%   opt_vecLen     - The optimal vector length that achieves the target deltaE value.
  
    % Define an anonymous function to compute the absolute difference between
    % the computed deltaE and the target deltaE value (1 JND).
    deltaE = @(d) abs(compute_deltaE(d, background_RGB, ref_RGB,...
        ref_Lab, vecDir, param) - stim.deltaE_1JND);

    % Set the lower and upper bounds for the vector length search.
    lb = 0; ub = 0.1;
    % Number of runs for the optimization to improve robustness
    N_runs  = 1;
    % Generate initial points for the optimization algorithm within the bounds.
    init    = rand(1,N_runs).*(ub-lb) + lb;
    % Set options for the optimization algorithm (fmincon).
    options = optimoptions(@fmincon, 'MaxIterations', 1e5, 'Display','off');
    % Preallocate arrays for storing results of each optimization run.
    [vecLen_n, deltaE_n] = deal(NaN(1, N_runs));

    % Perform the optimization using fmincon in a loop for each initial point.
    for n = 1:N_runs
        % Optimize to find the vector length that achieves the target deltaE.
        [vecLen_n(n), deltaE_n(n)] = fmincon(deltaE, init(n), ...
            [],[],[],[],lb,ub,[],options);
    end

    % Find the index of the run with the minimum deltaE value.
    [~,idx_min] = min(deltaE_n);
    % Select the optimal vector length that resulted in the closest deltaE to the target.
    opt_vecLen = vecLen_n(idx_min);
end
