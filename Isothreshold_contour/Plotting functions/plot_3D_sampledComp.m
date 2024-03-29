function plot_3D_sampledComp(ref_points, fitEllipsoid_unscaled, sampledComp,...
    fixedPlane, fixedPlaneVal, nPhiEllipse, nThetaEllipse, ellipse_2D, varargin)

    % Initialize an input parser for function arguments
    p = inputParser;
    p.addParameter('visualize_ellipsoid', true, @islogical);
    p.addParameter('visualize_samples', true, @islogical);
    p.addParameter('visualize_ellipse', false, @islogical);
    p.addParameter('visualize_ellipse_export_from_2Dsim', false, @islogical);
    p.addParameter('ellipse_2Dsim', [],@(x)(isnumeric(x)));
    p.addParameter('slc_grid_ref_dim1',1:length(ref_points),@(x)(isnumeric(x)));
    p.addParameter('slc_grid_ref_dim2',1:length(ref_points),@(x)(isnumeric(x)));
    p.addParameter('surf_alpha',0.3,@isnumeric);
    p.addParameter('samples_alpha',0.5,@isnumeric);
    p.addParameter('lineWidth_ellipse', 2, @isnumeric);
    p.addParameter('lineWidth_ellipse_2Dsim',1, @isnumeric);
    p.addParameter('markerSize_samples',10,@isnumeric);
    p.addParameter('default_viewing_angle',true,@islogical);
    p.addParameter('fontsize',15,@isnumeric);
    p.addParameter('title','',@ischar);
    p.addParameter('figPos', [0,0.1,0.425,0.8], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[30,30], @(x)(isnumeric(x)));
    p.addParameter('figName', 'Sampled comparison stimuli', @ischar);

    % Extract parsed input parameters
    parse(p, varargin{:});
    visualize_ellipsoid = p.Results.visualize_ellipsoid;
    visualize_samples   = p.Results.visualize_samples;
    visualize_ellipse   = p.Results.visualize_ellipse;
    visualize_ellipse_2Dsim = p.Results.visualize_ellipse_export_from_2Dsim;
    ellipse_2Dsim       = p.Results.ellipse_2Dsim;
    slc_grid_ref_dim1   = p.Results.slc_grid_ref_dim1;
    slc_grid_ref_dim2   = p.Results.slc_grid_ref_dim2;
    surf_alpha          = p.Results.surf_alpha;
    samples_alpha       = p.Results.samples_alpha;
    lw_ellipse          = p.Results.lineWidth_ellipse;
    lw_ellipse_2Dsim    = p.Results.lineWidth_ellipse_2Dsim;
    markerSize_samples  = p.Results.markerSize_samples;
    default_viewing_angle = p.Results.default_viewing_angle;
    fontsize   = p.Results.fontsize;
    ttl        = p.Results.title;
    saveFig    = p.Results.saveFig;
    figName    = p.Results.figName;
    figPos     = p.Results.figPos;
    paperSize  = p.Results.paperSize;

    % Determine the indices of the reference points based on the fixed 
    % plane specified ('R', 'G', or 'B' for different color channels)
    if strcmp(fixedPlane,'R')
        idx_x = find(fixedPlaneVal == ref_points);
    elseif strcmp(fixedPlane,'G')
        idx_y = find(fixedPlaneVal == ref_points);
    elseif strcmp(fixedPlane,'B')
        idx_z = find(fixedPlaneVal == ref_points);
    else
        error('Wrong plane name!')
    end
    nGridPts_dim1 = length(slc_grid_ref_dim1);
    nGridPts_dim2 = length(slc_grid_ref_dim2);
    ref_points_idx = 1:length(ref_points);

    figure
    tiledlayout(nGridPts_dim2,nGridPts_dim1,'TileSpacing','none');
    for j = nGridPts_dim2:-1:1
        jj = ref_points_idx(slc_grid_ref_dim2(j));
        for i = 1:nGridPts_dim1
            ii = ref_points_idx(slc_grid_ref_dim1(i));
            nexttile
        
            if strcmp(fixedPlane,'R')
                slc_ref = [fixedPlaneVal, ref_points(ii), ref_points(jj)];
                slc_gt = squeeze(fitEllipsoid_unscaled(idx_x,ii, jj,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(idx_x, ii,jj,:,:));
                slc_ellipse = squeeze(ellipse_2D(idx_x, ii,jj,1,:,:));
                if ~isempty(ellipse_2Dsim)
                    slc_ellipse_2Dsim = squeeze(ellipse_2Dsim(idx_x, ii,jj,1,:,:));
                end
            elseif strcmp(fixedPlane, 'G')
                slc_ref = [ref_points(ii),fixedPlaneVal, ref_points(jj)];
                slc_gt = squeeze(fitEllipsoid_unscaled(ii, idx_y, jj,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(ii, idx_y, jj,:,:));    
                slc_ellipse = squeeze(ellipse_2D(ii,idx_y,jj,2,:,:));
                if ~isempty(ellipse_2Dsim)
                    slc_ellipse_2Dsim = squeeze(ellipse_2Dsim(ii,idx_y,jj,2,:,:));
                end
            elseif strcmp(fixedPlane,'B')
                slc_ref = [ref_points(ii),ref_points(jj),fixedPlaneVal];
                slc_gt = squeeze(fitEllipsoid_unscaled(ii, jj, idx_z,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(ii,jj,idx_z,:,:));       
                slc_ellipse = squeeze(ellipse_2D(ii,jj,idx_z,3,:,:));
                if ~isempty(ellipse_2Dsim)
                    slc_ellipse_2Dsim = squeeze(ellipse_2Dsim(ii,jj,idx_z,3,:,:));
                end
            end

            hold on
            
            % Plot the 3D ellipsoid if enabled
            if visualize_ellipsoid
                h1 = surf(slc_gt_x,slc_gt_y,slc_gt_z,'FaceColor',slc_ref,...
                    'EdgeColor','none','FaceAlpha',surf_alpha);
                camlight headlight; lighting phong;
            end

            % Plot the 2D ellipse (derived from 3D simulations) if enabled and applicable
            if visualize_ellipse
                if strcmp(fixedPlane,'R')
                    h2 = plot3(slc_ref(1).*ones(size(slc_ellipse)), slc_ellipse(:,1), ...
                        slc_ellipse(:,2), 'Color',slc_ref,'LineWidth',lw_ellipse); 
                elseif strcmp(fixedPlane,'G')
                    h2 = plot3(slc_ellipse(:,1), slc_ref(2).*ones(size(slc_ellipse)),...
                        slc_ellipse(:,2), 'Color',slc_ref,'LineWidth',lw_ellipse); 
                else
                    h2 = plot3(slc_ellipse(:,1),  slc_ellipse(:,2), ...
                        slc_ref(3).*ones(size(slc_ellipse)),...
                        'Color',slc_ref,'LineWidth',lw_ellipse);                     
                end
            end
            hold on
            % Plot the 2D ellipse (derived from 2D simulations)
            if visualize_ellipse_2Dsim
                if strcmp(fixedPlane,'R')
                    h3 = plot3(slc_ref(1).*ones(size(slc_ellipse_2Dsim)),...
                        slc_ellipse_2Dsim(:,1), slc_ellipse_2Dsim(:,2),...
                        'Color','w','LineWidth',lw_ellipse_2Dsim,'LineStyle','--');         
                elseif strcmp(fixedPlane,'G')
                    h3 = plot3(slc_ellipse_2Dsim(:,1), ...
                        slc_ref(2).*ones(size(slc_ellipse_2Dsim)),...
                        slc_ellipse_2Dsim(:,2), 'Color','w','LineStyle','--',...
                        'LineWidth',lw_ellipse_2Dsim); 
                else
                    h3 = plot3(slc_ellipse_2Dsim(:,1),  slc_ellipse_2Dsim(:,2), ...
                        slc_ref(3).*ones(size(slc_ellipse_2Dsim)),...
                        'Color','w','LineWidth',lw_ellipse_2Dsim,'LineStyle','--');                     
                end
            end

            % Plot the 3D sampled points if enabled
            if visualize_samples
                scatter3(slc_rgb_comp(1,:), slc_rgb_comp(2,:), slc_rgb_comp(3,:),...
                    markerSize_samples, 'MarkerFaceColor', [0,0,0],...
                    'MarkerEdgeColor','none', 'MarkerFaceAlpha', samples_alpha);
            end

            
            % Setting the axes limits and appearance
            xlim(slc_ref(1)+[-0.025,0.025]);
            ylim(slc_ref(2)+[-0.025,0.025]);
            zlim(slc_ref(3)+[-0.025,0.025]); 
            axis square
            if strcmp(fixedPlane,'R'); xticks(slc_ref(1)); 
            else; xticks(slc_ref(1)+[-0.02,0,0.02]);
            end

            if strcmp(fixedPlane, 'G'); yticks(slc_ref(2));
            else;yticks(slc_ref(2)+[-0.02,0,0.02]);
            end

            if strcmp(fixedPlane, 'B'); zticks(slc_ref(3));
            else; zticks(slc_ref(3)+[-0.02,0,0.02]);
            end

             % Add legend and adjust viewing angle based on the fixed plane and user preferences
            if j == 1 && i == nGridPts_dim1
                if exist("h1","var") && exist("h2","var")
                    legend_list = {'Ground truth ellipsoid',...
                        ['Ground truth ellipse with fixed ',...
                        fixedPlane,' = ',num2str(fixedPlaneVal)]};
                else
                    if ~exist("h1","var")
                        legend_list = {['Ground truth ellipse with fixed ',...
                            fixedPlane,' = ',num2str(fixedPlaneVal)]};
                    else; legend_list = {'Ground truth ellipsoid'};
                    end
                end
                legend(legend_list,'Location','southeast'); legend boxoff; 
            end

            if strcmp(fixedPlane, 'R'); set(gca,'YDir','reverse'); end
            if ~default_viewing_angle
                if strcmp(fixedPlane,'R'); view(-90,0) %view from leftside
                elseif strcmp(fixedPlane,'G'); view(0,0);
                elseif strcmp(fixedPlane,'B'); view(0,90); 
                end
            else; view(-37.5, 30); grid on; camlight left; lighting phong;
            end
            set(gca,'FontSize',fontsize);
        end
    end
    sgtitle(ttl);
    set(gcf,'Units','Normalized','Position',figPos)

    if saveFig 
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        myFigDir = 'Simulation_FigFiles';
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        % Full path for the figure file
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
    end
end