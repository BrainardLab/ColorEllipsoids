clear all; close all; clc
%Load data grabbed from hpc and analyze the goodness of fits of ellipses

%we have a total of nDataFiles, each of which was generated by a different
%seed
nDataFiles     = 10;
plane_slc      = 'GB plane';
numSimPerCond  = 240;
samplingMethod = 'Random'; %'NearContour' or 'Random'
samplingJitter = 0.1;
samplingRange  = 0.025;
analysisDir    = getpref('ColorEllipsoids', 'ELPSAnalysis');
myDataDir      = 'ModelFitting_DataFiles/DataFiles_HPC';
intendedDir    = fullfile(analysisDir, myDataDir);
addpath(intendedDir);

flag_loadExistingModelComp = false;
for s = 1:nDataFiles
    disp(s)
    if strcmp(samplingMethod, 'NearContour')
        load(['Fits_isothreshold_',plane_slc,'_sim',num2str(numSimPerCond),...
            'perCond_sampling',samplingMethod,'_jitter',num2str(samplingJitter),...
            '_rng',num2str(s),'.mat'], 'E');
    elseif strcmp(samplingMethod, 'Random')
        load(['Fits_isothreshold_',plane_slc,'_sim',num2str(numSimPerCond),...
            'perCond_sampling',samplingMethod,'_range',num2str(samplingRange),...
            '_rng',num2str(s),'.mat'], 'E');      
    end
    param   = E{1}; 
    stim    = E{2}; 
    results = E{3}; 
    sim     = E{5}; 
    model   = E{6}; 
    fits    = E{7};
    modelComp.nLL_allRef(s) = min(fits.minVal);

    if flag_loadExistingModelComp
        addpath(fullfile(analysisDir, 'ModelComparison_DataFiles'));
        load(['Comparison_WishartM_vs_fullM_',plane_slc,'_sim', ...
            num2str(numSimPerCond), 'perCond_sampling',samplingMethod,...
            '_jitter',num2str(samplingJitter),'_rng1to',num2str(nDataFiles),'.mat']);
    else
        %model predictions (ellipses)
        modelComp.ngrid_bruteforce = 2e3;
        modelComp.vecLength = linspace(0,max(results.opt_vecLen(:))*1.2,...
            modelComp.ngrid_bruteforce);
    
        %for each reference stimulus
        for i = 1:length(sim.slc_ref) %stim.nGridPts_ref
            for j = 1:length(sim.slc_ref) %stim.nGridPts_ref
                %grab the reference stimulus's RGB
                rgb_ref_ij = sim.ref_points(sim.slc_ref(i),sim.slc_ref(j),:);
        
                [~, modelComp.WishartM_fitEllipse_unscaled(i,j,:,:,s), ...
                    ~,~,~, modelComp.WishartM_ellParam(i,j,:,s)] = ...
                    convert_Sig_2DisothresholdContour(rgb_ref_ij, sim.varying_RGBplane, ...
                    stim.grid_theta_xy, modelComp.vecLength, sim.pC_given_alpha_beta, ...
                    model.coeffs_chebyshev, fits.w_est_best, 'contour_scaler',...
                    results.contour_scaler, 'nSteps_bruteforce', modelComp.ngrid_bruteforce);
    
                modelComp.gtM_fitEllipse_unscaled(i,j,:,:,s) = squeeze(...
                    results.fitEllipse_unscaled(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane,i,j,:,:));
                modelComp.gtM_ellParam(i,j,:,s) = squeeze(...
                    results.ellParams(sim.slc_fixedVal_idx,...
                    sim.slc_RGBplane, i, j,:));
            end
        end
    end
end

%% compute mean and error bars
if ~flag_loadExistingModelComp
    errbar_lb_idx = 1;
    errbar_ub_idx = nDataFiles;
    for i = 1:length(sim.slc_ref)
        for j = 1:length(sim.slc_ref)
            %Wishart model
            axisRatio_WishartM = squeeze((1/modelComp.WishartM_ellParam(i,j,1,:))./(1/modelComp.WishartM_ellParam(i,j,2,:)));
            axisRatio_WishartM_sorted = sort(axisRatio_WishartM, 'ascend');
            modelComp.axisRatio_WishartM_mean(i,j) = mean(axisRatio_WishartM);
            modelComp.axisRatio_WishartM_lb(i,j) = axisRatio_WishartM_sorted(errbar_lb_idx);
            modelComp.axisRatio_WishartM_ub(i,j) = axisRatio_WishartM_sorted(errbar_ub_idx);
    
            rotAngle_WishartM = deg2rad(squeeze(modelComp.WishartM_ellParam(i,j,3,:)));
            rotAngle_WishartM_sorted = sort(rotAngle_WishartM, 'ascend');
            modelComp.rotAngle_WishartM_mean(i,j) = mean(rotAngle_WishartM_sorted);
            modelComp.rotAngle_WishartM_lb(i,j) = rotAngle_WishartM_sorted(errbar_lb_idx);
            modelComp.rotAngle_WishartM_ub(i,j) = rotAngle_WishartM_sorted(errbar_ub_idx); 
    
            [modelComp.unscaled_contour_CI_ub_WishartM(i,j,:,:),...
                modelComp.unscaled_contour_CI_lb_WishartM(i,j,:,:),...
                modelComp.contour_CI_ub_WishartM(i,j,:,:), modelComp.contour_CI_lb_WishartM(i,j,:,:)] = ...
                compute_contour_CI(squeeze(modelComp.WishartM_fitEllipse_unscaled(i,j,:,:,:)),...
                [stim.x_grid_ref(i,j), stim.y_grid_ref(i,j)]);
        end
    end
end

%%
plt.ttl = {'GB plane', 'RB plane', 'RG plane'};
if strcmp(sim.method_sampling,'NearContour')
    fig_str = [plt.ttl{sim.slc_RGBplane},...
        '_sim',num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter)];
elseif strcmp(sim.method_sampling, 'Random')
    fig_str  = [plt.ttl{sim.slc_RGBplane},...
        '_sim',num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end))];
end

plot_2D_isothreshold_contour(stim.x_grid_ref, stim.y_grid_ref, ...
    results.fitEllipse(sim.slc_fixedVal_idx,...
    sim.slc_RGBplane,:,:,:,:),sim.slc_fixedVal,...
    'slc_x_grid_ref',sim.slc_ref,...
    'slc_y_grid_ref',sim.slc_ref,...
    'WishartEllipses_contour_CI',{modelComp.contour_CI_lb_WishartM, modelComp.contour_CI_ub_WishartM},...
    'IndividualEllipses_contour_CI',{},...
    'rgb_background',false,...
    'xlabel',plane_slc(1),...
    'ylabel',plane_slc(2),...
    'refColor',[0,0,0],...
    'EllipsesColor',[178,34,34]./255,...
    'WishartEllipsesColor',[76,153,0]./255,...
    'ExtrapEllipsesColor',[0.5,0.5,0.5],...
    'figPos',[0,0.1,0.35,0.4],...
    'subTitle', {sprintf(['Comparison of the predictions for the iso-threshold contours \nin ',...
        plane_slc, ' between the Wishart model and the ground truth'])},...
    'figName', ['Comparison_IsothresholdContours_WishartM_vs_gtM_',fig_str, '_rng',num2str(s)],...
    'saveFig',false);

%% visualize
%ratio between major and minor axis
plot_comparison_WishartM_vs_fullM(squeeze(modelComp.gtM_ellParam(:,:,2,1))./squeeze(modelComp.gtM_ellParam(:,:,1,1)),...
    modelComp.axisRatio_WishartM_mean,modelComp.axisRatio_WishartM_lb,...
    modelComp.axisRatio_WishartM_ub,[],[],...
    'xlim',[1,7],'ylim',[1,7],'x_ref',stim.grid_ref,'y_ref',stim.grid_ref,...
    'xlabel',sprintf('The ground-truth ratio of major vs. minor axis'),...
    'ylabel',sprintf('The ratio of major vs. minor axis\n computed based on the Wishart Process Model'),...
    'figPos',[0,0.1,0.3,0.5],'paperSize',[25,25],'saveFig',false,...
    'figName',['Comparison_axisRatio_fullM_vs_gtM_',fig_str, '_rng1to',num2str(nDataFiles)]);

%rotation angle
plot_comparison_WishartM_vs_fullM(squeeze(deg2rad(modelComp.gtM_ellParam(:,:,3,1))),...
    modelComp.rotAngle_WishartM_mean,modelComp.rotAngle_WishartM_lb,...
    modelComp.rotAngle_WishartM_ub,[],[],...
    'xlim',[0,pi/2],'ylim',[0,pi/2],'x_ref',stim.grid_ref,'y_ref',stim.grid_ref,...
    'xticks',0:pi/8:pi/2, 'yticks',0:pi/8:pi/2, ...
    'xticklabels',{'$0$','$\pi/8$','$\pi/4$','$3\pi/8$','$\pi/2$'},...
    'yticklabels',{'$0$','$\pi/8$','$\pi/4$','$3\pi/8$','$\pi/2$'},...
    'LatexInterpreter',true,...
    'xlabel',sprintf('The ground-truth rotation (in radians)'),...
    'ylabel',sprintf('The rotation (in radians)\n computed based on the Wishart Process Model'),...
    'figPos',[0,0.1,0.3,0.5],'paperSize',[25,25],'saveFig',false,...
    'figName',['Comparison_rotAngle_WishartM_vs_gtM_',fig_str, '_rng1to',num2str(nDataFiles)]);

%% save data
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'ModelComparison_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end

if strcmp(sim.method_sampling, 'NearContour')
    fileName = ['Comparison_WishartM_vs_gtM_',plane_slc,'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_jitter',num2str(sim.random_jitter),'_rng1to',num2str(nDataFiles),'.mat'];
elseif strcmp(sim.method_sampling, 'Random')
    fileName = ['Comparison_WishartM_vs_gtM_',plane_slc,'_sim',...
        num2str(sim.nSims), 'perCond_sampling',sim.method_sampling,...
        '_range',num2str(sim.range_randomSampling(end)),'_rng1to',num2str(nDataFiles),'.mat'];
end
outputName = fullfile(outputDir, fileName);
save(outputName,'modelComp');




