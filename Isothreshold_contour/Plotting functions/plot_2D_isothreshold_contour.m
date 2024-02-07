function plot_2D_isothreshold_contour(x_grid_ref, y_grid_ref, fitEllipse,...
    fixed_RGBvec, varargin)

    %the number of curves we are plotting
    nFrames = size(fitEllipse,1);
    %throw an error is 
    assert(length(fixed_RGBvec) == nFrames);
    
    p = inputParser;
    p.addParameter('visualizeRawData', false, @islogical);
    p.addParameter('rgb_contour', [], @(x)(isnumeric(x)));
    p.addParameter('figTitle', {'GB plane', 'RB plane', 'RG plane'}, @(x)(ischar(x)));
    p.addParameter('normalizedFigPos', [0,0.1,0.55,0.4], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('figName', 'Isothreshold_contour', @ischar);

    parse(p, varargin{:});
    visualizeRawData = p.Results.visualizeRawData;
    rgb_contour   = p.Results.rgb_contour;
    figTitle      = p.Results.figTitle;
    saveFig       = p.Results.saveFig;
    figName       = p.Results.figName;
    figPos        = p.Results.normalizedFigPos;
    %throw an error is 
    assert(size(rgb_contour,1) == nFrames);

    nPlanes = size(fitEllipse,2);
    nGridPts_ref_x = length(x_grid_ref);
    nGridPts_ref_y = length(y_grid_ref);
    
    x = [0; 0; 1; 1];
    y = [1; 0; 0; 1];

    figure
    for l = 1:nFrames        
        for p = 1:nPlanes
            %define colormap
            colormapMatrix = NaN(4,nPlanes); 
            colormapMatrix(:,p) = fixed_RGBvec(l);
            indices = setdiff(1:nPlanes,p);
            colormapMatrix(:,indices(1)) = x;
            colormapMatrix(:,indices(2)) = y;

            subplot(1, nPlanes, p)
            patch('Vertices', [x,y],'Faces', [1 2 3 4], 'FaceVertexCData', ...
                colormapMatrix, 'FaceColor', 'interp'); hold on
            scatter(x_grid_ref(:), y_grid_ref(:), 20,'white','Marker','+');
        
            for i = 1:nGridPts_ref_x
                for j = 1:nGridPts_ref_y
                    %visualize the individual thresholds 
                    if visualizeRawData
                        scatter(squeeze(rgb_contour(l,p,i,j,:,1)),...
                            squeeze(rgb_contour(l,p,i,j,:,2)),10,...
                            'o','filled','MarkerEdgeColor',0.5.*ones(1,3),...
                            'MarkerFaceColor',0.5.*ones(1,3));
                    end
                    %visualize the best-fitting ellipse
                    plot(squeeze(fitEllipse(l,p,i,j,:,1)),...
                        squeeze(fitEllipse(l,p,i,j,:,2)),...
                        'white-','lineWidth',1.5)
                end
            end
            xlim([0,1]); ylim([0,1]); axis square; hold off
            xticks(0:0.2:1); yticks(0:0.2:1);
            title(figTitle{p})
            xlabel(figTitle{p}(1)); ylabel(figTitle{p}(2));
        end
        sgtitle(['The fixed other plane = ',num2str(fixed_RGBvec(l))]);
        set(gcf,'Units','normalized','Position',figPos);
        set(gcf,'PaperUnits','centimeters','PaperSize',[40 12]);
        if saveFig && nFrames > 1
            if l == 1; gif([figName, '.gif'])
            else; gif
            end
        end
        pause(1)
    end
    if saveFig && nFrames == 1 %1 frame
        set(gcf,'PaperUnits','centimeters','PaperSize',[40 12]);
        saveas(gcf, [figName, '.pdf']);
    end
end
