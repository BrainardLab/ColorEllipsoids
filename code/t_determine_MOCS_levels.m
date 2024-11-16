clear all; close all; clc
%load file name
cal_path = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/';
matfileName = 'Transformation_btw_color_spaces.mat';
outputName = fullfile(cal_path,matfileName);
load(outputName, 'DELL_11092024_withoutGammaCorrection');

M_2DWToDKLPlane = DELL_11092024_withoutGammaCorrection.M_2DWToDLKPlane;

%% load data
pd_path = ['/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/Meta_analysis/',...
    'Pilot_DataFiles/Pilot_MOCS/'];
nSets_MOCS = 8;
nLevels = 12;
nTrials = 120;
stimulus_DKLplane = NaN(nSets_MOCS, nTrials, 2);
response = NaN(nSets_MOCS, nTrials);
file_names = {'MOCS_sub3_set1.csv', 'MOCS_sub3_set2.csv', 'MOCS_sub4_set1.csv', 'MOCS_sub3_set4.csv', ...
    'MOCS_sub4_set2.csv', 'MOCS_sub3_set6.csv', 'MOCS_sub4_set3.csv', 'MOCS_sub3_set8.csv'};
for n = 1:nSets_MOCS
    % Load csv file e.g., MOCS_sub1_set#.csv where # is n
    file_name = file_names{n};
    data_n = readmatrix(fullfile(pd_path, file_name));

    % Extract the first two columns from the csv file
    stimulus_W_n = data_n(:, 1:2);

    %convert the two columns from W space to DKL space
    stimulus_W_n_full = horzcat(stimulus_W_n, ones(nTrials, 1));
    stimulus_DKLplane_n = M_2DWToDKLPlane * stimulus_W_n_full';

    %store in the matrix
    stimulus_DKLplane(n,:,:) = stimulus_DKLplane_n(1:2,:)';

    % Extract the response from the csv file
    resp_n = data_n(:, 3);

    % Store it
    response(n, :) = resp_n';
end

%% plot psychometric functions
[unique_stimulus_varyingDim, sorted_unique_stimulus_varyingDim] = deal(NaN(nSets_MOCS, nLevels));
varyingDim_MOCS = NaN(1, nSets_MOCS);
varyingDKL_dim = [2,1];
fig = figure;
ax_handles = gobjects(1, nSets_MOCS);  % Preallocate an array to store axis handles

for idx = 1:nSets_MOCS
    ax_handles(idx) = subplot(2, nSets_MOCS/2, idx);
    stimulus_slc = squeeze(stimulus_DKLplane(idx,:,:));
    if min(abs(stimulus_slc(:,1) - mean(stimulus_slc(:,1), 1))) < 1e-5
        varyingDim_MOCS(idx) = 2;
    else
        varyingDim_MOCS(idx) = 1;
    end
    unique_stimulus_varyingDim(idx,:) = unique(stimulus_slc(:, varyingDim_MOCS(idx)));
    sorted_unique_stimulus_varyingDim(idx,:) = sort(unique_stimulus_varyingDim(idx,:));
    assert(length(unique_stimulus_varyingDim(idx,:)) == nLevels, ...
        'The number of levels do not match the number of unique stimuli!');

    for ss = 1:length(unique_stimulus_varyingDim(idx,:))
        idx_ss = find(unique_stimulus_varyingDim(idx,ss) == stimulus_slc(:, varyingDim_MOCS(idx)));
        resp_org(ss) = sum(response(idx,idx_ss))./length(idx_ss);
    end
    scatter(ax_handles(idx), unique_stimulus_varyingDim(idx,:), resp_org, ...
        'k','filled', 'MarkerEdgeColor','white'); hold on

    ylim([0,1]); xlim([min(unique_stimulus_varyingDim(idx,:))-1e-4, max(unique_stimulus_varyingDim(idx,:))+1e-4]);
    yticks([0.33, 0.67, 1]);
    xticks(round([sorted_unique_stimulus_varyingDim(idx,1),...
        sorted_unique_stimulus_varyingDim(idx,(end-1):end)],4));
end

%% fit a weibull
weibullfunc = @(alpha, beta, x) 1/3 + (1-1/3).*(1- exp(-(x./alpha).^beta));
estimated_weibullP = NaN(nSets_MOCS, 2);
predicted_xLevels = 1000;
% Initial guess for [alpha, beta]
initial_params = [1, 1];  % Adjust this initial guess based on your expected parameter values

% Set bounds for [alpha, beta] (optional)
lb = [1e-3, 1e-1];   % lower bounds for alpha and beta
ub = [5, 5];  % upper bounds for alpha and beta

[predicted_x, predicted_probC] = deal(NaN(nSets_MOCS, predicted_xLevels));
for n = 1:nSets_MOCS
    %organize data (1st row: stimulus; 2nd row: response, 1 or 0)
    D_n = vertcat(squeeze(stimulus_DKLplane(n, :, varyingDim_MOCS(n))) - ...
        min(unique_stimulus_varyingDim(n,:)), response(n,:));
    
    %negative log likelihood function
    nLL = @(p) -sum(D_n(2,:).*log(weibullfunc(p(1), p(2), D_n(1,:))) + ...
                (1-D_n(2,:)).*log(1-weibullfunc(p(1), p(2), D_n(1,:))));

    % Minimize the negative log-likelihood function
    [estimated_params, ~] = fmincon(@(p) nLL(p), initial_params, [], [], [], [], lb, ub);

    % Store the estimated parameters for the current set
    estimated_weibullP(n, :) = estimated_params;

    %create a finely sampled x
    predicted_x(n,:) = linspace(0, max(unique_stimulus_varyingDim(n,:)) - ...
        min(unique_stimulus_varyingDim(n,:)), predicted_xLevels);

    predicted_probC(n,:) = weibullfunc(estimated_params(1), estimated_params(2), predicted_x(n,:));

    plot(ax_handles(n), predicted_x(n,:) + min(unique_stimulus_varyingDim(n,:)),...
        predicted_probC(n,:), 'k-', 'lineWidth',2);
end
set(gcf,'Units','normalized','Position',[0,0,0.4, 0.2]);
set(gcf,'PaperSize',[15,5]);
saveas(fig, fullfile(pd_path, 'MOCS_pilots.pdf'));


