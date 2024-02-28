function plot_3D_RGBplanes(plane_points, colormapMatrix, varargin)
% PLOT_3D_RGBPLANES Plots the GB, RB, or RG plane within a 3D RGB color cube.
% For each plane, the unselected color dimension is fixed at a specific value.
% If the fixed value is a vector, the function generates a series of plots
% that can be saved as frames in a GIF for visualization.
%
% Inputs:
%   plane_points     - Cell array containing the coordinates for plotting
%                      each plane in the RGB cube.
%   colormapMatrix   - Cell array of colormaps for each set of plane points.
%   varargin         - Variable input arguments including:
%       'nGridPts'          - Number of grid points for each plane. Default is 100.
%       'ref_points'        - Cell array of reference points to be plotted on each plane.
%       'figTitle'          - Cell array of titles for each plot. Default titles are for
%                             GB, RB, and RG planes.
%       'normalizedFigPos'  - Array specifying the figure's normalized position on screen.
%       'saveFig'           - Logical flag to save the figure(s) as a PDF or GIF. Default is false.
%       'paperSize'         - Array specifying the size of the saved figure in centimeters.
%       'figName'           - Name of the file when saving the figure.
%

    %the number of curves we are plotting
    nFrames = length(plane_points);
    %throw an error is 
    assert(length(colormapMatrix) == nFrames);

    % Validate input arguments and set default values
    p = inputParser;
    p.addParameter('nGridPts',100, @(x) floor(x)==x);
    p.addParameter('ref_points',{}, @(x)(iscell(x) && (numel(x) == nFrames)));
    p.addParameter('visualize_surfacePlane',true);
    p.addParameter('visualize_refStimuli',true);
    p.addParameter('figTitle', {'GB plane', 'RB plane', 'RG plane'}, @(x)(ischar(x)));
    p.addParameter('normalizedFigPos', [0,0.1,0.7,0.4], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[30,12], @(x)(isnumeric(x)));
    p.addParameter('figName', 'RGB_cube', @ischar);

    % Extract parsed input parameters
    parse(p, varargin{:});
    nGridPts   = p.Results.nGridPts;
    ref_points = p.Results.ref_points;
    visualize_surfacePlane = p.Results.visualize_surfacePlane;
    visualize_refStimuli = p.Results.visualize_refStimuli;
    figTitle   = p.Results.figTitle;
    saveFig    = p.Results.saveFig;
    figName    = p.Results.figName;
    figPos     = p.Results.normalizedFigPos;
    paperSize  = p.Results.paperSize;

    % Determine the number of planes to plot based on the input
    nPlanes = length(plane_points{1});
    grid    = linspace(0, 1,nGridPts);
    [x_grid, y_grid] = meshgrid(grid, grid);

    % Initialize figure for plotting
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
        
            %Plot the specified plane within the RGB cube
            %p = 1: GB plane; p = 2: RB plane; p = 3: RG plane
            if visualize_surfacePlane
                surf(plane_points{l}{p}(:,:,1),plane_points{l}{p}(:,:,2),...
                    plane_points{l}{p}(:,:,3), colormapMatrix{l}{p},...
                    'EdgeColor','none');
            end

            % If reference points are provided, plot them
            if visualize_refStimuli
                if length(size(ref_points{l}{p})) == 3
                    scatter3(ref_points{l}{p}(:,:,1),ref_points{l}{p}(:,:,2),...
                        ref_points{l}{p}(:,:,3), 20,'k','Marker','+'); 
                elseif length(size(ref_points{l}{p})) == 4
                    for n = 1:5
                        scatter3(ref_points{l}{p}(:,:,n,1),ref_points{l}{p}(:,:,n,2),...
                            ref_points{l}{p}(:,:,n,3), 20,'k','Marker','+'); 
                    end
                end
            end
            hold off

            % Set plot limits and labels
            xlim([0,1]); ylim([0,1]); zlim([0,1]); axis equal
            xlabel('R'); ylabel('G'); zlabel('B')
            xticks(0:0.2:1); yticks(0:0.2:1); zticks(0:0.2:1);

            % Set title for each subplot
            title(figTitle{p});
        end

        % Adjust figure properties
        set(gcf,'Units','normalized','Position',figPos)
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);

        % Save figure as gif if requested
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

        % Pause to ensure the plot updates properly in the loop
        pause(0.5)
    end

    %save figure as pdf if requested
    if saveFig && nFrames == 1 
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

