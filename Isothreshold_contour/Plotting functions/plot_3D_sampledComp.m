function plot_3D_sampledComp(ref_points, fitEllipsoid_unscaled, sampledComp,...
    fixedPlane, fixedPlaneVal, nPhiEllipse, nThetaEllipse, varargin)

    % Validate input arguments and set default values
    p = inputParser;
    p.addParameter('slc_grid_ref_dim1',1:length(ref_points),@(x)(isnumeric(x)));
    p.addParameter('slc_grid_ref_dim2',1:length(ref_points),@(x)(isnumeric(x)));
    p.addParameter('surf_alpha',0.3,@isnumeric);
    p.addParameter('default_viewing_angle',true,@islogical);
    p.addParameter('fontsize',15,@isnumeric);
    p.addParameter('title','',@ischar);
    p.addParameter('figPos', [0,0.1,0.425,0.8], @(x)(isnumeric(x) && numel(x)==4));
    p.addParameter('saveFig', false, @islogical);
    p.addParameter('paperSize',[30,30], @(x)(isnumeric(x)));
    p.addParameter('figName', 'Sampled comparison stimuli', @ischar);

    % Extract parsed input parameters
    parse(p, varargin{:});
    slc_grid_ref_dim1 = p.Results.slc_grid_ref_dim1;
    slc_grid_ref_dim2 = p.Results.slc_grid_ref_dim2;
    surf_alpha = p.Results.surf_alpha;
    default_viewing_angle = p.Results.default_viewing_angle;
    fontsize   = p.Results.fontsize;
    ttl        = p.Results.title;
    saveFig    = p.Results.saveFig;
    figName    = p.Results.figName;
    figPos     = p.Results.figPos;
    paperSize  = p.Results.paperSize;

    %
    if strcmp(fixedPlane,'R')
        idx_x = find(fixedPlaneVal == ref_points);
    elseif strcmp(fixedPlane,'G')
        idx_y = find(fixedPlaneVal == ref_points);
    elseif strcmp(fixedPlane,'B')
        idx_z = find(fixedPlaneVal == ref_points);
    else
        error('Wrong plane name!')
    end
    nGridPts_dim1 = length(slc_grid_ref_dim1);
    nGridPts_dim2 = length(slc_grid_ref_dim2);
    ref_points_idx = 1:length(ref_points);

    figure
    tiledlayout(nGridPts_dim2,nGridPts_dim1,'TileSpacing','none');
    for j = nGridPts_dim2:-1:1
        jj = ref_points_idx(slc_grid_ref_dim2(j));
        for i = 1:1:nGridPts_dim1
            ii = ref_points_idx(slc_grid_ref_dim1(i));
            nexttile
        
            if strcmp(fixedPlane,'R')
                slc_ref = [fixedPlaneVal, ref_points(ii), ref_points(jj)];
                slc_gt = squeeze(fitEllipsoid_unscaled(idx_x,ii, jj,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(idx_x, ii,jj,:,:));
            elseif strcmp(fixedPlane, 'G')
                slc_ref = [ref_points(ii),fixedPlaneVal, ref_points(jj)];
                slc_gt = squeeze(fitEllipsoid_unscaled(ii, idx_y, jj,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(ii, idx_y, jj,:,:));          
            elseif strcmp(fixedPlane,'B')
                slc_ref = [ref_points(ii),ref_points(jj),fixedPlaneVal];
                slc_gt = squeeze(fitEllipsoid_unscaled(ii, jj, idx_z,:,:));
                slc_gt_x = reshape(slc_gt(:,1), [nPhiEllipse, nThetaEllipse]);
                slc_gt_y = reshape(slc_gt(:,2), [nPhiEllipse, nThetaEllipse]);
                slc_gt_z = reshape(slc_gt(:,3), [nPhiEllipse, nThetaEllipse]);
                slc_rgb_comp = squeeze(sampledComp(ii,jj,idx_z,:,:));                 
            end
            
            scatter3(slc_rgb_comp(1,:), slc_rgb_comp(2,:), slc_rgb_comp(3,:),10,...
                'MarkerFaceColor', [0,0,0],'MarkerEdgeColor','none',...
                'MarkerFaceAlpha',0.3);hold on
            surf(slc_gt_x,slc_gt_y,slc_gt_z,'FaceColor',slc_ref,...
                'EdgeColor','none','FaceAlpha',surf_alpha)
            camlight right; lighting phong;
            xlim(slc_ref(1)+[-0.025,0.025]);
            ylim(slc_ref(2)+[-0.025,0.025]);
            zlim(slc_ref(3)+[-0.025,0.025]); 
            axis square
            xticks(slc_ref(1)+[-0.02,0,0.02]);
            yticks(slc_ref(2)+[-0.02,0,0.02]);
            zticks(slc_ref(3)+[-0.02,0,0.02]);

            if strcmp(fixedPlane, 'R'); set(gca,'YDir','reverse'); end
            %set(gca,'XDir','reverse');
            % if i == 3 && j == 3; xlabel('R'); ylabel('G'); zlabel('B');end
            if ~default_viewing_angle
                if strcmp(fixedPlane,'R'); view(-90,0) %view from leftside
                elseif strcmp(fixedPlane,'G'); view(90,0);
                elseif strcmp(fixedPlane,'B'); view(0,90); 
                end
            end
            set(gca,'FontSize',fontsize);
        end
    end
    sgtitle(ttl);
    set(gcf,'Units','Normalized','Position',figPos)

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