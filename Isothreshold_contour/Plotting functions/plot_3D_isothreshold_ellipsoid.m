function plot_3D_isothreshold_ellipsoid(x_grid_ref, y_grid_ref, z_grid_ref,...
    fitEllipsoid, nTheta, nPhi, varargin)

    % Validate input arguments and set default values
    p = inputParser;
    p.addParameter('slc_x_grid_ref',1:length(x_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('slc_y_grid_ref',1:length(y_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('slc_z_grid_ref',1:length(z_grid_ref),@(x)(isnumeric(x)));
    p.addParameter('visualize_ref',true);
    p.addParameter('visualize_ellipsoids',true);
    p.addParameter('visualize_thresholdPoints',false);
    p.addParameter('threshold_points',[],@isnumeric);
    p.addParameter('ms_ref',100,@isnumeric);
    p.addParameter('lw_ref',2,@isnumeric);
    p.addParameter('color_ref_rgb',[],@(x)(isnumeric(x)));
    p.addParameter('color_surf',[],@(x)(isnumeric(x)));
    p.addParameter('color_threshold',[],@(x)(isnumeric(x)));
    p.addParameter('surf_alpha', 0.5, @(x)(isnumeric(x)));
    p.addParameter('azimuthAngle', -37.5, @isnumeric);   
    p.addParameter('elevationAngle', 30, @isnumeric); 
    p.addParameter('fontsize',15,@isnumeric);
    p.addParameter('view_angle',[-37.5, 30],@(x)(isnumeric(x)));
    p.addParameter('flag_flipX',false, @islogical);
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
    visualize_thresholdPoints = p.Results.visualize_thresholdPoints;
    threshold_points = p.Results.threshold_points;
    ms_ref           = p.Results.ms_ref;
    lw_ref           = p.Results.lw_ref;
    color_ref_rgb    = p.Results.color_ref_rgb;
    color_surf       = p.Results.color_surf;
    color_threshold  = p.Results.color_threshold;
    surf_alpha       = p.Results.surf_alpha;
    azimuthAngle     = p.Results.azimuthAngle;
    elevationAngle   = p.Results.elevationAngle;
    fontsize   = p.Results.fontsize;
    view_angle = p.Results.view_angle;
    flag_flipX = p.Results.flag_flipX;
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
                %visualize the reference stimuli (indicated by fixational
                %corss)
                if visualize_ref
                    if isempty(color_ref_rgb)
                        cmap_ijk = [x_grid_ref(ii), y_grid_ref(jj), z_grid_ref(kk)];
                    else
                        cmap_ijk = color_ref_rgb;
                    end
                    scatter3(cmap_ijk(1),cmap_ijk(2),cmap_ijk(3),ms_ref, ...
                        cmap_ijk,'+','lineWidth',lw_ref); hold on
                end
                
                %visualize the best-fitting ellipsoids
                if visualize_ellipsoids
                    if isempty(color_surf)
                        cmap_ijk = [x_grid_ref(ii), y_grid_ref(jj), z_grid_ref(kk)];
                    else
                        cmap_ijk = color_surf;
                    end
                    ell_ijk = squeeze(fitEllipsoid(ii,jj,kk,:,:));
                    ell_ijk_x = ell_ijk(:,1); 
                    ell_ijk_x_reshape = reshape(ell_ijk_x, [nPhi, nTheta]);
                    
                    ell_ijk_y = ell_ijk(:,2); 
                    ell_ijk_y_reshape = reshape(ell_ijk_y, [nPhi, nTheta]);
                    
                    ell_ijk_z = ell_ijk(:,3); 
                    ell_ijk_z_reshape = reshape(ell_ijk_z, [nPhi, nTheta]);
    
                    surf(ell_ijk_x_reshape, ell_ijk_y_reshape, ...
                        ell_ijk_z_reshape,'FaceColor', cmap_ijk,...
                        'EdgeColor','none','FaceAlpha', surf_alpha); hold on
                end

                %visualize the individual thresholds at all chromatic
                %directions
                if visualize_thresholdPoints && ~isempty(threshold_points)
                    if isempty(color_threshold)
                        cmap_ijk = [x_grid_ref(ii), y_grid_ref(jj), z_grid_ref(kk)];
                    else
                        cmap_ijk = color_threshold;
                    end
                    scatter3(squeeze(threshold_points(ii,jj,kk,:,:,1)),...
                        squeeze(threshold_points(ii,jj,kk,:,:,2)),...
                        squeeze(threshold_points(ii,jj,kk,:,:,3)),...
                        10,cmap_ijk,'filled','MarkerFaceAlpha',0.5); 
                end
                % axis vis3d equal;
                axis equal; axis square;
            end
        end
    end
    view(azimuthAngle, elevationAngle);
    xlim([0,1]); ylim([0,1]); zlim([0,1])
    xticks(sort([0,1,x_grid_ref_trunc])); yticks(sort([0,1,y_grid_ref_trunc])); 
    zticks(sort([0,1,z_grid_ref_trunc])); 
    xlabel('R'); ylabel('G'); zlabel('B');
    % set(gca,'YDir','reverse');
    set(gca,'FontSize', fontsize);
    set(gcf,'Units','normalized','Position',figPos);

    light_handle = camlight('right'); lighting phong
    view(view_angle(1), view_angle(2));
    if flag_flipX; set(gca, 'XDir', 'reverse'); end
    if saveFig
        set(gcf,'PaperUnits','centimeters','PaperSize',paperSize);
        analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
        myFigDir = 'Simulation_FigFiles';
        outputDir = fullfile(analysisDir, myFigDir);
        if ~exist(outputDir, 'dir')
            mkdir(outputDir);
        end
        figFilePath = fullfile(outputDir, [figName, '.pdf']);
        saveas(gcf, figFilePath);
        %set(gcf, 'Renderer', 'opengl');  % Switch to OpenGL renderer
        %print(gcf, figFilePath, '-dpdf', '-vector', '-r300');
    end

    % % Initial camlight
    % for az = 0:5:360
    %     view(az, -37.5);
    %     delete(light_handle);
    %     light_handle = camlight('right');  % Adjust the camlight to headlight first
    %     lighting phong
    %     drawnow;  % Update the figure window
    %     pause(0.05);  % Pause for 0.05 seconds between updates
    % end
end