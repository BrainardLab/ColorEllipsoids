function plot_3D_RGBplanes(plane_points, colormapMatrix, varargin)

    %the number of curves we are plotting
    nFrames = length(plane_points);
    %throw an error is 
    assert(length(colormapMatrix) == nFrames);

    p = inputParser;
    p.addParameter('nGridPts',100, @(x) floor(x)==x);
    p.addParameter('ref_points',{}, @(x)(iscell(x) && (numel(x) == nFrames)));
    p.addParameter('figTitle', {'GB plane', 'RB plane', 'RG plane'}, @(x)(ischar(x)));
    p.addParameter('normalizedFigPos', [0,0.1,0.7,0.4], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[30,12], @(x)(isnumeric(x)));
    p.addParameter('figName', 'RGB_cube', @ischar);

    parse(p, varargin{:});
    nGridPts   = p.Results.nGridPts;
    ref_points = p.Results.ref_points;
    figTitle   = p.Results.figTitle;
    saveFig    = p.Results.saveFig;
    figName    = p.Results.figName;
    figPos     = p.Results.normalizedFigPos;
    paperSize  = p.Results.paperSize;

    nPlanes = length(plane_points{1});
    grid    = linspace(0, 1,nGridPts);
    [x_grid, y_grid] = meshgrid(grid, grid);

    %number of selected planes
    figure
    for l = 1:nFrames
        for p = 1:nPlanes
            subplot(1,nPlanes,p)
            %floor of the cube
            surf(x_grid, y_grid, zeros(nGridPts, nGridPts),...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.01); hold on
            %ceiling of the cube 
            surf(x_grid, y_grid, ones(nGridPts, nGridPts),...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.01);
            %walls of the cube
            surf(x_grid, ones(nGridPts, nGridPts), y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(x_grid, zeros(nGridPts, nGridPts), y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(zeros(nGridPts, nGridPts), x_grid, y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            surf(ones(nGridPts, nGridPts), x_grid, y_grid,...
                'FaceColor','k','EdgeColor','none','FaceAlpha',0.05);
            %wall edges
            plot3(zeros(1,nGridPts),zeros(1,nGridPts), grid,'k-');
            plot3(zeros(1,nGridPts),ones(1,nGridPts), grid,'k-');
            plot3(ones(1,nGridPts),zeros(1,nGridPts), grid,'k-');
            plot3(ones(1,nGridPts),ones(1,nGridPts), grid,'k-');
        
            %p = 1: GB plane; p = 2: RB plane; p = 3: RG plane
            surf(plane_points{l}{p}(:,:,1),plane_points{l}{p}(:,:,2),...
                plane_points{l}{p}(:,:,3), colormapMatrix{l}{p},...
                'EdgeColor','none');
            %reference points
            scatter3(ref_points{l}{p}(:,:,1),ref_points{l}{p}(:,:,2),...
                ref_points{l}{p}(:,:,3), 20,'k','Marker','+'); hold off
            xlim([0,1]); ylim([0,1]); zlim([0,1]); axis equal
            xlabel('R'); ylabel('G'); zlabel('B')
            xticks(0:0.2:1); yticks(0:0.2:1); zticks(0:0.2:1);
            title(figTitle{p});
        end
        set(gcf,'Units','normalized','Position',figPos)
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        if saveFig && nFrames > 1
            if l == 1
                set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
                analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
                myFigDir = 'Simulation_FigFiles';
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
        pause(0.5)
    end
    if saveFig && nFrames == 1 %1 frame
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

