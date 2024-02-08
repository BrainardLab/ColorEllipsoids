function plot_multiHeatmap(M, varargin)
    p = inputParser;
    p.addParameter('cmap',"summer", @isstring);

    parse(p, varargin{:});
    cmap   = p.Results.cmap;

    nRows = size(M,3);
    nCols = size(M,4);
    figure
    for r = 1:nRows
        for c = 1:nCols
            colormap(cmap)
            subplot(nRows, nCols, c+nCols*(r-1))
            imagesc(M(:,:,r,c));
            xticks([]); yticks([]); axis square;
        end
    end
end