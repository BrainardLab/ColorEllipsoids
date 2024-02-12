function plot_2D_sampledComp(grid_ref_x, grid_ref_y, rgb_comp, ...
    varying_RGBplane, method_sampling, varargin)
    p = inputParser;
    p.addParameter('groundTruth',[],@(x)(isnumeric(x)));
    p.addParameter('modelPredictions',[],@(x)(isnumeric(x)));
    p.addParameter('responses',[],@(x)(isnumeric(x)));
    p.addParameter('xbds', [-0.025, 0.025], @(x)(isnumeric(x) && length(x)==2));
    p.addParameter('ybds', [-0.025, 0.025], @(x)(isnumeric(x) && length(x)==2));
    p.addParameter('nFinerGrid',50,@isnumeric);
    p.addParameter('EllipsesColor',[178,34,34]./255, @(x)(isnumeric(x)));
    p.addParameter('WishartEllipsesColor',[76,153,0]./255, @(x)(isnumeric(x)));
    p.addParameter('marker1',".",@ischar);
    p.addParameter('marker0','*',@ischar);
    p.addParameter('markerColor1', [173,216,230]./255, @(x)(isnumeric(x) && length(x)==3));
    p.addParameter('markerColor0', [255,179,138]./255, @(x)(isnumeric(x) && length(x)==3));
    p.addParameter('figPos', [0, 0.1,0.415,0.7], @(x)(isnumeric(x) && length(x)==4));
    p.addParameter('saveFig',false,@islogical);
    p.addParameter('figName','Sampled comparison stimuli', @ischar);

    parse(p, varargin{:});
    groundTruth      = p.Results.groundTruth;
    modelPredictions = p.Results.modelPredictions;
    resp_binary      = p.Results.responses;
    xbds             = p.Results.xbds;
    ybds             = p.Results.ybds;
    nFinerGrid       = p.Results.nFinerGrid;
    mc_ellipse       = p.Results.EllipsesColor;
    mc_ellipseW      = p.Results.WishartEllipsesColor;
    mk1              = p.Results.marker1;
    mk0              = p.Results.marker0;
    mc1              = p.Results.markerColor1;
    mc0              = p.Results.markerColor0;
    figPos           = p.Results.figPos;
    saveFig          = p.Results.saveFig;
    figName          = p.Results.figName;

    nGrid_x = length(grid_ref_x);
    nGrid_y = length(grid_ref_y);

    figure
    cmap = colormap("gray"); colormap(flipud(cmap))
    t = tiledlayout(nGrid_x,nGrid_y,'TileSpacing','none');
    for i = nGrid_x:-1:1  %row: B
        for j = 1:nGrid_y %column: G
            x_axis = linspace(xbds(1),xbds(2),nFinerGrid)+grid_ref_x(j);
            y_axis = linspace(xbds(1),ybds(2),nFinerGrid)+grid_ref_y(i);
    
            nexttile
            %simulated trials
            if strcmp(method_sampling,'NearContour')
                h = histcounts2(squeeze(rgb_comp(i,j,varying_RGBplane(2),:)),...
                    squeeze(rgb_comp(i,j, varying_RGBplane(1),:)),...
                    y_axis,x_axis);
                imagesc(x_axis, y_axis, h); axis square; hold on;
                set(gca,'YDir','normal') 
            elseif strcmp(method_sampling, 'Random')
                idx_1 = find(resp_binary(i,j,:)==1);
                idx_0 = find(resp_binary(i,j,:)==0);
                scatter(squeeze(rgb_comp(i,j,varying_RGBplane(1),idx_1)),...
                    squeeze(rgb_comp(i,j, varying_RGBplane(2),idx_1)),mk1,...
                    'MarkerFaceColor',mc1,'MarkerEdgeColor',mc1,'MarkerFaceAlpha',0.5); hold on
                scatter(squeeze(rgb_comp(i,j,varying_RGBplane(1),idx_0)),...
                    squeeze(rgb_comp(i,j, varying_RGBplane(2),idx_0)),mk0,...
                    'MarkerFaceColor',mc0,'MarkerEdgeColor',mc0,'MarkerFaceAlpha',0.5);
            else
                error('Unidentified sampling method');
            end
    
            % ground truth
            if ~isempty(groundTruth)
                h1 = plot(squeeze(groundTruth(i,j,:,1)),squeeze(groundTruth(i,j,:,2)),...
                     'Color',mc_ellipse,'lineStyle','--','lineWidth',1);    
            end

            % fits
            if ~isempty(modelPredictions)
                h2 = plot(squeeze(modelPredictions(i,j,:,1)),...
                     squeeze(modelPredictions(i,j,:,2)),...
                     'Color',mc_ellipseW,'lineStyle','-','lineWidth',1); 
            end
            
            hold off; box on;
            xlim([x_axis(1), x_axis(end)]); ylim([y_axis(1), y_axis(end)]);
    
            if j == 1; yticks(grid_ref_y(i));
            else; yticks([]);end
    
            if i == 1; xticks(grid_ref_x(j));
            else; xticks([]);end

            if exist('h2','var') && exist('h1','var')
                if i == 1 && j == nGrid_y
                    legend([h1, h2], {'Ground truth','Wishart model predictions'},...
                        'Location','southeast'); 
                    legend boxoff
                end
            end

            set(gca,'FontSize',12)
        end
    end
    set(gcf,'Units','Normalized','Position',figPos);
    set(gcf,'PaperUnits','centimeters','PaperSize',[35 35]);
    if saveFig
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        if ~exist('h2','var'); myFigDir = 'Simulation_FigFiles';
        else; myFigDir = 'ModelFitting_FigFiles'; end
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        % Full path for the figure file
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
    end
end