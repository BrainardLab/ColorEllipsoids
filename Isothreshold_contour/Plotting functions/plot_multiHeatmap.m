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
    p.addParameter('D', [], @(x)(isnumeric(x)));

    parse(p, varargin{:});
    permute_M = p.Results.permute_M;
    cmap   = p.Results.cmap;
    X      = p.Results.X;
    Y      = p.Results.Y;
    x_ticks= p.Results.x_ticks;
    y_ticks= p.Results.y_ticks;
    sgttl  = p.Results.sgttl;
    D      = p.Results.D; 

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
        end
    end
    sgtitle(sgttl);
end