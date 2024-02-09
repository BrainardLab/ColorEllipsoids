clear all; close all; clc

%% load isothreshold contours simulated based on CIELAB
load('Isothreshold_contour_CIELABderived.mat', 'D');
param   = D{1};
stim    = D{2};
results = D{3};
plt     = D{4};

%% First create a cube and select the RG, the RB and the GB planes
sim.slc_fixedVal     = 0.5; %the level of the fixed plane
sim.slc_fixedVal_idx = find(stim.fixed_RGBvec == sim.slc_fixedVal);
sim.slc_RGBplane     = 1; %GB plane with a fixed R value
sim.varying_RGBplane = setdiff(1:3, sim.slc_RGBplane);
sim.plane_points     = param.plane_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.ref_points       = stim.ref_points{sim.slc_fixedVal_idx}{sim.slc_RGBplane};
sim.background_RGB   = stim.background_RGB(:,sim.slc_fixedVal_idx);
sim.fitA             = squeeze(results.fitA(sim.slc_fixedVal_idx,sim.slc_RGBplane,:,:,:,:));

%% Simulate data given the iso-threshold contours
sim.alpha               = 1.1729;
sim.beta                = 1.2286;
sim.pC_given_alpha_beta = ComputeWeibTAFC(stim.deltaE_1JND,sim.alpha,sim.beta);%0.8;
sim.nSims_perDir        = 15; %5 x 16 = 80; 15 x 16 = 240; 50 x 16 = 800
sim.nSims               = sim.nSims_perDir * (stim.numDirPts -1);
sim.range_randomSampling = [-0.025, 0.025];
sim.method_sampling     = 'Random';

%for each reference stimulus
for i = 1:stim.nGridPts_ref
    for j = 1:stim.nGridPts_ref
        %grab the reference stimulus's RGB
        rgb_ref_pij = squeeze(sim.ref_points(i,j,:));
        %convert it to Lab
        [ref_Lab_lpij, ~, ~] = convert_rgb_lab(param.B_monitor, ...
            sim.background_RGB, param.T_cones, param.M_LMSToXYZ, rgb_ref_pij);
        sim.ref_Lab(i,j,:) = ref_Lab_lpij;
        
        %simulate comparison stimulus
        if strcmp(sim.method_sampling, 'NearContour')
            opt_vecLen_ij = squeeze(results.opt_vecLen(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane, i, j, :, :));
            rgb_contour_cov = squeeze(results.rgb_contour_cov(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane, i, j, :, :,:));
            sim.rgb_comp(i,j,:,:) = sample_rgb_comp_along2DChromDir(...
                rgb_ref_pij(sim.varying_RGBplane), sim.varying_RGBplane,...
                sim.slc_RGBplane, stim.grid_theta_xy, sim.nSims_perDir,...
                opt_vecLen_ij, rgb_contour_cov);
        elseif strcmp(sim.method_sampling, 'Random')
            sim.rgb_comp(i,j,:,:) = sample_rgb_comp_2Drandom(...
                rgb_ref_pij(sim.varying_RGBplane), sim.varying_RGBplane,...
                sim.slc_fixedVal, sim.range_randomSampling, sim.nSims);
            % sim.rgb_comp(i,j,:,:) = draw_rgb_comp_random(sim, ...
            %     rgb_ref_pij(sim.varying_RGBplane), [-0.025, 0.025]);
        end
        
        %simulate binary responses
        for n = 1:sim.nSims
            [sim.lab_comp(i,j,:,n), ~, ~] = convert_rgb_lab(param.B_monitor, ...
                sim.background_RGB, param.T_cones, param.M_LMSToXYZ, ...
                squeeze(sim.rgb_comp(i,j,:,n)));
            sim.deltaE(i,j,n) = norm(squeeze(sim.lab_comp(i,j,:,n)) - ref_Lab_lpij);
            sim.probC(i,j,n) = ComputeWeibTAFC(sim.deltaE(i,j,n), ...
                sim.alpha, sim.beta);
        end
        sim.resp_binary(i,j,:) = binornd(1, squeeze(sim.probC(i,j,:)), [sim.nSims, 1]);
    end
end

%% visualize samples
groundTruth_slc = squeeze(results.fitEllipse_unscaled(sim.slc_fixedVal_idx,...
    sim.slc_RGBplane,:,:,:,:));
ttlName = ['Fitted_Isothreshold_contour_w',sim.method_sampling,...
        'Samples_',plt.ttl{sim.slc_RGBplane},'_sim',num2str(sim.nSims), 'perCond']; 
if strcmp(sim.method_sampling, 'NearContour')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...
        'groundTruth', groundTruth_slc,...
        'saveFig',false,...
        'title',ttlName)
elseif strcmp(sim.method_sampling, 'Random')
    plot_2D_sampledComp(stim.grid_ref, stim.grid_ref, sim.rgb_comp, ...
        sim.varying_RGBplane, sim.method_sampling,...
        'groundTruth', groundTruth_slc,...
        'responses',sim.resp_binary,...
        'saveFig',false, ...
        'title',ttlName)    
end

%% save the data
save(['Sims_isothreshold_',strtrim(plt.ttl{sim.slc_RGBplane}),'_sim',...
    num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,'.mat'],'sim');


