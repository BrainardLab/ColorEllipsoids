function plot_comparison_WishartM_vs_fullM(x, y, y_lb, y_ub, x_lb, x_ub, varargin)
    
    p = inputParser;
    p.addParameter('x_ref',[],@(x)(isnumeric(x)));
    p.addParameter('y_ref',[],@(x)(isnumeric(x)));
    p.addParameter('xlim',[],@(x)(isnumeric(x) && length(x)==2));
    p.addParameter('ylim',[],@(x)(isnumeric(x) && length(x)==2));
    p.addParameter('xlabel','',@ischar);
    p.addParameter('ylabel','',@ischar);
    p.addParameter('xticks',[],@(x)(isnumeric(x)));
    p.addParameter('yticks',[],@(x)(isnumeric(x)));
    p.addParameter('xticklabels',{},@iscell);
    p.addParameter('yticklabels',{},@iscell);
    p.addParameter('LatexInterpreter',false,@islogical);
    p.addParameter('figPos', [0,0.1,0.55,0.4], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[40,20], @(x)(isnumeric(x)));
    p.addParameter('figName', 'Comparison_WishartM_vs_fullM', @ischar);

    parse(p, varargin{:});
    x_ref            = p.Results.x_ref;
    y_ref            = p.Results.y_ref;
    x_bds            = p.Results.xlim;
    y_bds            = p.Results.ylim;
    xlbl             = p.Results.xlabel;
    ylbl             = p.Results.ylabel;
    xts              = p.Results.xticks;
    yts              = p.Results.yticks;
    xtslbl           = p.Results.xticklabels;
    ytslbl           = p.Results.yticklabels;
    flag_latex       = p.Results.LatexInterpreter;
    saveFig          = p.Results.saveFig;
    figName          = p.Results.figName;
    paperSize        = p.Results.paperSize;
    figPos           = p.Results.figPos;

    %throw errors if the followings do not match 
    assert(size(x,1) == size(y,1) && size(x,2) == size(y,2),...
        "x and y have to be of the same size!");

    figure
    %identity line
    plot(x_bds, x_bds,'Color',[0.5,0.5,0.5]); hold on
    for i = 1:size(x,1)
        for j = 1:size(x,2)
            cmap_ij = [0.5,x_ref(j),y_ref(i)];
            %with both horizontal and vertical errorbars
            if ~isempty(x_lb) && ~isempty(x_ub)
                errorbar(x(i,j), y(i,j), y(i,j)-y_lb(i,j), y_ub(i,j)-y(i,j),...
                    x(i,j)-x_lb(i,j),x_ub(i,j)-x(i,j),'Marker','square',...
                    'MarkerEdgeColor',cmap_ij,'MarkerFaceColor',cmap_ij,...
                    'Color',cmap_ij,'MarkerSize',10,'LineWidth',1.5); hold on
            %with just vertical errorbars
            else
                errorbar(x(i,j), y(i,j), y(i,j)-y_lb(i,j), y_ub(i,j)-y(i,j),...
                    'Marker','square', 'MarkerEdgeColor',cmap_ij,...
                    'MarkerFaceColor',cmap_ij, 'Color',cmap_ij,...
                    'MarkerSize',10,'LineWidth',1.5); hold on
            end
        end
    end
    axis equal; xlim(x_bds); ylim(y_bds);grid on
    xlabel(xlbl); if ~isempty(xts); xticks(xts);end
    if ~isempty(xtslbl);xticklabels(xtslbl);end
    ylabel(ylbl); if ~isempty(yts); yticks(yts);end
    if ~isempty(ytslbl);yticklabels(ytslbl);end
    if flag_latex; set(gca,'TickLabelInterpreter','Latex'); end
    set(gca,'FontSize',15);

    if saveFig 
        set(gcf,'Units','normalized','Position',figPos);
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        myFigDir = 'ModelComparison_FigFiles';
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        % Full path for the figure file
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
    end

end