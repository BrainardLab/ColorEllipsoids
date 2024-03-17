function plot_multiHeatmap(M, varargin)
    %all the heatmaps share the same x and y axes
    assert(length(size(M))==4, "The input matrix has to be 4D!");

    p = inputParser;
    p.addParameter('permute_M',false,@islogical);
    p.addParameter('cmap',"summer", @isstring);
    p.addParameter('X', [], @(x)(isnumeric(x)));
    p.addParameter('Y', [], @(x)(isnumeric(x)));
    p.addParameter('x_ticks',[],@(x)(isnumeric(x)));
    p.addParameter('y_ticks',[],@(x)(isnumeric(x)));
    p.addParameter('sgttl',"",@isstring);
    p.addParameter('colorbar_on', false, @islogical);
    p.addParameter('D', [], @(x)(isnumeric(x)));
    p.addParameter('figPos', [0, 0.1,0.415,0.7], @(x)(isnumeric(x) && length(x)==4));
    p.addParameter('saveFig',false,@islogical);
    p.addParameter('figName','MultiHeatmaps', @ischar);

    parse(p, varargin{:});
    permute_M = p.Results.permute_M;
    cmap      = p.Results.cmap;
    X         = p.Results.X;
    Y         = p.Results.Y;
    x_ticks   = p.Results.x_ticks;
    y_ticks   = p.Results.y_ticks;
    sgttl     = p.Results.sgttl;
    colorbar_on = p.Results.colorbar_on;
    D         = p.Results.D; 
    figPos    = p.Results.figPos;
    saveFig   = p.Results.saveFig;
    figName   = p.Results.figName;

    if permute_M; M = permute(M, [3,4,1,2]); end

    nRows = size(M,1);
    nCols = size(M,2);
    if ~isempty(D)
        assert(size(D,1)==nRows && size(D,2)==nCols && size(D,3)==2)
    end

    figure
    for r = 1:nRows
        for c = 1:nCols
            colormap(cmap)
            subplot(nRows, nCols, c+nCols*(r-1))

            if isempty(X) || isempty(Y)
                imagesc(squeeze(M(r,c,:,:)));
            else
                imagesc(X,Y,squeeze(M(r,c,:,:)));
            end

            %scatterplot
            if ~isempty(D)
                hold on; scatter(D(r,c,1), D(r,c,2),15,'mo','filled');
            end
            xticks(x_ticks); yticks(y_ticks); 
            axis square;
            if colorbar_on; colorbar; clim([0,1]); end
        end
    end
    sgtitle(sgttl);
    set(gcf,'Units','Normalized','Position',figPos);
    set(gcf,'PaperUnits','centimeters','PaperSize',[35 35]);
    if saveFig
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        myFigDir = 'WishartPractice_FigFiles'; 
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        % Full path for the figure file
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
    end
end