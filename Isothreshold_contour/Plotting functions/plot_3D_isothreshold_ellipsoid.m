function plot_3D_isothreshold_ellipsoid(x_grid_ref, y_grid_ref, z_grid_ref,...
    fitEllipsoid, nTheta, nPhi, varargin)

    % Validate input arguments and set default values
    p = inputParser;
    p.addParameter('slc_x_grid_ref',1:length(x_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('slc_y_grid_ref',1:length(y_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('slc_z_grid_ref',1:length(z_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('visualize_ref',true);
    p.addParameter('visualize_ellipsoids',true);
    p.addParameter('ms_ref',100,@isnumeric);
    p.addParameter('lw_ref',2,@isnumeric);
    p.addParameter('color_surf',[0.8,0.8,0.8],@(x)(isnumeric(x)));
    p.addParameter('fontsize',15,@isnumeric);
    p.addParameter('normalizedFigPos', [0,0.1,0.3,0.5], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[30,30], @(x)(isnumeric(x)));
    p.addParameter('figName', 'Isothreshold_ellipsoids', @ischar);

    % Extract parsed input parameters
    parse(p, varargin{:});
    slc_x_grid_ref = p.Results.slc_x_grid_ref;
    slc_y_grid_ref = p.Results.slc_y_grid_ref;
    slc_z_grid_ref = p.Results.slc_z_grid_ref;
    visualize_ref = p.Results.visualize_ref;
    visualize_ellipsoids = p.Results.visualize_ellipsoids;
    ms_ref     = p.Results.ms_ref;
    lw_ref     = p.Results.lw_ref;
    color_surf = p.Results.color_surf;
    fontsize   = p.Results.fontsize;
    saveFig    = p.Results.saveFig;
    figName    = p.Results.figName;
    figPos     = p.Results.normalizedFigPos;
    paperSize  = p.Results.paperSize;

    %selected ref points 
    x_grid_ref_trunc = x_grid_ref(slc_x_grid_ref);
    y_grid_ref_trunc = y_grid_ref(slc_y_grid_ref);
    z_grid_ref_trunc = z_grid_ref(slc_z_grid_ref);

    nGridPts_ref_x = length(x_grid_ref_trunc);
    nGridPts_ref_y = length(y_grid_ref_trunc);
    nGridPts_ref_z = length(z_grid_ref_trunc);

    figure; 
    for i = 1:nGridPts_ref_x
        ii = slc_x_grid_ref(i);
        for j = 1:nGridPts_ref_y
            jj = slc_y_grid_ref(j);
            for k = 1:nGridPts_ref_z
                kk = slc_z_grid_ref(k);
                if visualize_ref
                    cmap_ijk = [x_grid_ref(ii), y_grid_ref(jj), z_grid_ref(kk)];
                    scatter3(cmap_ijk(1),cmap_ijk(2),cmap_ijk(3),ms_ref, ...
                        cmap_ijk,'+','lineWidth',lw_ref); hold on
                end

                if visualize_ellipsoids
                    ell_ijk = squeeze(fitEllipsoid(ii,jj,kk,:,:));
                    ell_ijk_x = ell_ijk(:,1); 
                    ell_ijk_x_reshape = reshape(ell_ijk_x, [nPhi, nTheta]);
                    
                    ell_ijk_y = ell_ijk(:,2); 
                    ell_ijk_y_reshape = reshape(ell_ijk_y, [nPhi, nTheta]);
                    
                    ell_ijk_z = ell_ijk(:,3); 
                    ell_ijk_z_reshape = reshape(ell_ijk_z, [nPhi, nTheta]);
    
                    surf(ell_ijk_x_reshape, ell_ijk_y_reshape, ...
                        ell_ijk_z_reshape,'FaceColor', color_surf,...
                        'EdgeColor','none','FaceAlpha',0.5); hold on
                end
                % axis vis3d equal;
                axis equal; axis square;
            end
        end
    end
    camlight right; lighting phong
    xlim([0,1]); ylim([0,1]); zlim([0,1])
    xticks(sort([0,1,x_grid_ref_trunc])); yticks(sort([0,1,y_grid_ref_trunc])); 
    zticks(sort([0,1,z_grid_ref_trunc])); 
    xlabel('R'); ylabel('G'); zlabel('B');
    set(gca,'FontSize', fontsize);
    set(gcf,'Units','normalized','Position',figPos);

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