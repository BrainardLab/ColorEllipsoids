function plot_2D_isothreshold_contour(x_grid_ref, y_grid_ref, fitEllipse,...
    fixed_RGBvec, varargin)

    %the number of curves we are plotting
    nFrames = size(fitEllipse,1);
    %throw an error is 
    assert(length(fixed_RGBvec) == nFrames);
    
    p = inputParser;
    p.addParameter('slc_x_grid_ref',1:length(x_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('slc_y_grid_ref',1:length(y_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('visualizeRawData', false, @islogical);
    p.addParameter('WishartEllipses',[],@(x)(isnumeric(x)));
    p.addParameter('WishartEllipses_contour_CI',{},@iscell);
    p.addParameter('IndividualEllipses_contour_CI',{},@iscell);
    p.addParameter('ExtrapEllipses',[],@(x)(isnumeric(x)));
    p.addParameter('rgb_contour', [], @(x)(isnumeric(x)));
    p.addParameter('rgb_background',true, @islogical);
    p.addParameter('subTitle', {'GB plane', 'RB plane', 'RG plane'}, @iscell);
    p.addParameter('refColor',[0,0,0],@(x)(isnumeric(x)));
    p.addParameter('EllipsesColor',[1,1,1], @(x)(isnumeric(x)));
    p.addParameter('WishartEllipsesColor',[76,153,0]./255, @(x)(isnumeric(x)));
    p.addParameter('ExtrapEllipsesColor',[0.5,0.5,0.5],@(x)(isnumeric(x)));
    p.addParameter('EllipsesLine','--',@ischar);
    p.addParameter('WishartEllipsesLine','-',@ischar);
    p.addParameter('ExtrapEllipsesLine',':',@ischar);
    p.addParameter('xlabel','',@ischar);
    p.addParameter('ylabel','',@ischar);
    p.addParameter('figPos', [0,0.1,0.55,0.4], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[40,20], @(x)(isnumeric(x)));
    p.addParameter('figName', 'Isothreshold_contour', @ischar);

    parse(p, varargin{:});
    slc_x_grid_ref   = p.Results.slc_x_grid_ref;
    slc_y_grid_ref   = p.Results.slc_y_grid_ref;
    visualizeRawData = p.Results.visualizeRawData;
    WishartEllipses  = p.Results.WishartEllipses;
    WishartEllipses_contour_CI = p.Results.WishartEllipses_contour_CI;
    IndividualEllipses_contour_CI = p.Results.IndividualEllipses_contour_CI;
    ExtrapEllipses   = p.Results.ExtrapEllipses;
    rgb_contour      = p.Results.rgb_contour;
    rgb_background   = p.Results.rgb_background;
    subTitle         = p.Results.subTitle;
    mc0              = p.Results.refColor;
    mc1              = p.Results.EllipsesColor;
    mc2              = p.Results.WishartEllipsesColor;
    mc3              = p.Results.ExtrapEllipsesColor;
    ls1              = p.Results.EllipsesLine;
    ls2              = p.Results.WishartEllipsesLine;
    ls3              = p.Results.ExtrapEllipsesLine;
    xlbl             = p.Results.xlabel;
    ylbl             = p.Results.ylabel;
    saveFig          = p.Results.saveFig;
    figName          = p.Results.figName;
    paperSize        = p.Results.paperSize;
    figPos           = p.Results.figPos;
    %throw an error is 
    if ~isempty(rgb_contour); assert(size(rgb_contour,1) == nFrames); end

    nPlanes = size(fitEllipse,2);
    nGridPts_ref_x = length(x_grid_ref(slc_x_grid_ref));
    nGridPts_ref_y = length(y_grid_ref(slc_x_grid_ref));
    
    x = [0; 0; 1; 1];
    y = [1; 0; 0; 1];

    figure
    for l = 1:nFrames        
        for p = 1:nPlanes
            subplot(1, nPlanes, p); 

            %define colormap
            if rgb_background
                colormapMatrix = NaN(4,nPlanes); 
                colormapMatrix(:,p) = fixed_RGBvec(l);
                indices = setdiff(1:nPlanes,p);
                colormapMatrix(:,indices(1)) = x;
                colormapMatrix(:,indices(2)) = y;
                patch('Vertices', [x,y],'Faces', [1 2 3 4], 'FaceVertexCData', ...
                    colormapMatrix, 'FaceColor', 'interp'); hold on
            end
            
            for i = 1:length(x_grid_ref)
                for j = 1:length(y_grid_ref)
                    %visualize the individual thresholds 
                    if visualizeRawData
                        scatter(squeeze(rgb_contour(l,p,i,j,:,1)),...
                            squeeze(rgb_contour(l,p,i,j,:,2)),20,...
                            'o','filled','MarkerEdgeColor',0.5.*ones(1,3),...
                            'MarkerFaceColor',0.5.*ones(1,3)); hold on
                    end
                    %visualize the ground truth ellipse
                    h1 = plot(squeeze(fitEllipse(l,p,i,j,:,1)),...
                        squeeze(fitEllipse(l,p,i,j,:,2)),...
                        'LineStyle',ls1,'Color',mc1,'lineWidth',1.5); hold on

                    if ~isempty(IndividualEllipses_contour_CI)
                        contour_CI_lb_ij = squeeze(IndividualEllipses_contour_CI{1}(i,j,:,:));
                        contour_CI_ub_ij = squeeze(IndividualEllipses_contour_CI{2}(i,j,:,:));
                        patch([contour_CI_lb_ij(1,1), contour_CI_ub_ij(:,1)', ...
                            fliplr(contour_CI_lb_ij(:,1)')], [contour_CI_lb_ij(1,2),...
                            contour_CI_ub_ij(:,2)', fliplr(contour_CI_lb_ij(:,2)')],...
                            [0,0,0],'FaceAlpha',0.2,'EdgeColor','none');
                    end
                end
            end

            for i = 1:nGridPts_ref_x
                for j = 1:nGridPts_ref_y
                    %cross for reference stimulus
                    scatter(x_grid_ref(slc_x_grid_ref(i),slc_y_grid_ref(j)),...
                        y_grid_ref(slc_x_grid_ref(i), slc_y_grid_ref(j)), 20,mc0,'Marker','+'); hold on;

                    %visualize the model-predicted ellipses
                    if ~isempty(WishartEllipses)
                        h2 = plot(squeeze(WishartEllipses(l,p,i,j,:,1)),...
                            squeeze(WishartEllipses(l,p,i,j,:,2)),...
                            'lineStyle',ls2,'Color',mc2,'lineWidth',1.5);
                    end

                    %confidence interval
                    if ~isempty(WishartEllipses_contour_CI)
                        contour_CI_lb_ij = squeeze(WishartEllipses_contour_CI{1}(i,j,:,:));
                        contour_CI_ub_ij = squeeze(WishartEllipses_contour_CI{2}(i,j,:,:));
                        h2 = patch([contour_CI_lb_ij(1,1), contour_CI_ub_ij(:,1)', ...
                            fliplr(contour_CI_lb_ij(:,1)')], [contour_CI_lb_ij(1,2),...
                            contour_CI_ub_ij(:,2)', fliplr(contour_CI_lb_ij(:,2)')],...
                            mc2,'FaceAlpha',0.3,'EdgeColor','none');
                    end
                end
            end

            %visualize ellipses at extrapolated locations
            if ~isempty(ExtrapEllipses)
                nrows = size(ExtrapEllipses,3);
                ncols = size(ExtrapEllipses, 4);
                for i = 1:nrows
                    for j = 1:ncols
                        h3 = plot(squeeze(ExtrapEllipses(l,p,i,j,:,1)),...
                            squeeze(ExtrapEllipses(l,p,i,j,:,2)),...
                            'lineStyle',ls3,'Color',mc3,'lineWidth',1.5);
                    end
                end
            end

            xlim([0,1]); ylim([0,1]); axis square; hold off
            xticks(0:0.2:1); yticks(0:0.2:1);
            title(subTitle{p}); 
            if length(subTitle) > 1
                xlabel(subTitle{p}(1)); ylabel(subTitle{p}(2));
            else; xlabel(xlbl); ylabel(ylbl); 
            end
            if exist('h2','var')
                if exist('h3','var')
                    legend([h1, h2, h3], {'Ground truth',...
                        'Wishart model predictions','Extrapolations'},...
                        'Location','southeast'); 
                else
                    legend([h1, h2], {'Ground truth','Wishart model predictions'},...
                        'Location','southeast'); 
                end
                legend boxoff
            end
        end
        box off;
        if nFrames > 1; sgtitle(['The fixed other plane = ',num2str(fixed_RGBvec(l))]); end
        set(gcf,'Units','normalized','Position',figPos);
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        if saveFig && nFrames > 1
            if l == 1
                analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
                if ~exist('h2','var'); myFigDir = 'Simulation_FigFiles';
                else; myFigDir = 'ModelFitting_FigFiles'; end
                outputDir = fullfile(analysisDir, myFigDir);
                if ~exist(outputDir, 'dir')
                    mkdir(outputDir);
                end
                % Full path for the figure file
                figFilePath = fullfile(outputDir, [figName, '.gif']);
                gif(figFilePath)
            else; gif
            end
        end
        pause(1)
    end
    if saveFig && nFrames == 1 %1 frame
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        if ~exist('h2','var'); myFigDir = 'Simulation_FigFiles';
        else; myFigDir = 'ModelFitting_FigFiles'; end
        if ~isempty(WishartEllipses_contour_CI) || ~isempty(IndividualEllipses_contour_CI)
            myFigDir = 'ModelComparison_FigFiles'; 
        end
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        % Full path for the figure file
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
    end
end
