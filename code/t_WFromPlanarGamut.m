% t_WFromPlanarGamut
%
% Try to figure out how we take (e.g.) the isoluminant plane
% and put a [-1,1] x [-1,1] coordinate system on the region
% of it that is within the gamut of a monitor.

%% Initialize
clear all; close all;

%% Retrieve the correct calibration file
whichCalFile = 'DELL_11072024_withoutGammaCorrection.mat';
whichCalNumber = 1;
nDeviceBits = 14; %doesn't have to be the true color depth; we can go higher
whichCones = 'ss2';
cal_path = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/';
cal = LoadCalFile(whichCalFile,whichCalNumber,cal_path);

%% Cone cal object
calObjCones = ObjectToHandleCalOrCalStruct(cal);

% Make the bit depth correct as far as the calibration file goes.
nDeviceLevels = 2^nDeviceBits;
CalibrateFitGamma(calObjCones, nDeviceLevels);
nPrimaries = calObjCones.get('nDevices');

% Set gamma mode. A value of 2 was used in the experiment
%   gammaMode == 0 - search table using linear interpolation via interp1.
%   gammaMode == 1 - inverse table lookup.  Fast but less accurate.
%   gammaMode == 2 - exhaustive search
gammaMode = 2;
SetGammaMethod(calObjCones,gammaMode);

% Set wavelength support.
Scolor = calObjCones.get('S');

% Cone fundamentals.
switch (whichCones)
    case 'asano'
        psiParamsStruct.coneParams = DefaultConeParams('cie_asano');
        psiParamsStruct.coneParams.fieldSizeDegrees = 2;
        psiParamsStruct.coneParams.ageYears = 30;
        T_cones = ComputeObserverFundamentals(psiParamsStruct.coneParams,Scolor);
    case 'ss2'
        load T_cones_ss2
        T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,Scolor);
        T_cones1 = T_cones;
end
SetSensorColorSpace(calObjCones,T_cones,Scolor);

%% XYZ cal object
calObjXYZ = ObjectToHandleCalOrCalStruct(cal);

% Get gamma correct
CalibrateFitGamma(calObjXYZ, nDeviceLevels);
SetGammaMethod(calObjXYZ,gammaMode);

%% XYZ
USE1931XYZ = false;
if (USE1931XYZ)
    load T_xyz1931.mat
    T_xyz = 683*SplineCmf(S_xyz1931,T_xyz1931,Scolor);
else
    load T_xyzCIEPhys2.mat
    T_xyz = 683*SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,Scolor);
end
T_Y = T_xyz(2,:);
SetSensorColorSpace(calObjXYZ,T_xyz,Scolor);

%% LMS <-> RGB
M_RGBToLMS = T_cones*calObjXYZ.cal.P_device;
M_LMSTORGB = inv(M_RGBToLMS);

%% Compute ambient
ambientLMS = SettingsToSensor(calObjCones,[0 0 0]');
ambientXYZ = SettingsToSensor(calObjXYZ,[0 0 0']');

%% Compute the background, taking quantization into account
% The calculations here account for display quantization.
SPECIFIEDBG = false;
if (SPECIFIEDBG)
    bgxyYTarget = [0.31, 0.31, 100]';
    bgXYZTarget = xyYToXYZ(bgxyYTarget);
    bgPrimary = SettingsToPrimary(calObjXYZ,SensorToSettings(calObjXYZ,bgXYZTarget));
else
    bgPrimary = SettingsToPrimary(calObjCones,PrimaryToSettings(calObjCones,[0.5 0.5 0.5]'));
end
bgXYZ = PrimaryToSensor(calObjXYZ,bgPrimary);
bgxyY = XYZToxyY(bgXYZ);
bgLMS = PrimaryToSensor(calObjCones,bgPrimary);

% Print basic info and report on monitor
fprintf('\nCalibration file %s, calibration number %d, calibration date %s\n', ...
    whichCalFile,whichCalNumber,calObjXYZ.cal.describe.date);
fprintf('\nBackground x,y = %0.4f, %0.4f\n',bgxyY(1),bgxyY(2));
fprintf('Background Y = %0.2f cd/m2, ambient %0.3f cd/m2\n',bgXYZ(2),ambientXYZ(2));

%% Basic transformation matrices.  ComputeDKL_M() does the work.
% Get matrix that transforms between incremental
% cone coordinates and DKL coordinates 
% (Lum, RG, S).
[M_ConeIncToDKL,~] = ComputeDKL_M(bgLMS,T_cones,T_Y);
M_DKLToConeInc = inv(M_ConeIncToDKL);

% Cone increment to cone contrast and back
M_ConeContrastToConeInc = diag(bgLMS);
M_ConeIncToConeContrast = diag(1./bgLMS);

%% Find incremental cone directions corresponding to DKL  directions.
lumConeInc = M_DKLToConeInc*[1 0 0]';
rgConeInc = M_DKLToConeInc*[0 1 0]';
sConeInc = M_DKLToConeInc*[0 0 1]';

% These directions should (and do last I checked) have unit pooled cone contrast,
% the way that the matrix M is scaled by ComputeDKL_M.
lumPooled = norm(lumConeInc ./ bgLMS);
rgPooled = norm(rgConeInc ./ bgLMS);
sPooled = norm(sConeInc ./ bgLMS);
fprintf(['Pooled cone contrast for unit DKL directions with initial scaling:',...
    '%0.3g %0.3g %0.3g\n'], lumPooled, rgPooled, sPooled);

%% Max contrast
% initialize
nAngles = 1000;
[gamutSettings, gamutLMS, gamutContrast, gamutDKL] = deal(NaN(3, nAngles));
vectorLengthGamutContrast = NaN(1, nAngles);

% Find maximum in gamut contrast for a set of color directions.
% This calculation does not worry about quantization.  It is
% done for the main case.  
theAngles = linspace(0,2*pi,nAngles);
gamut_bg_primary = NaN(3, nAngles);

%color space involved:
%
for aa = 1:nAngles
    % Get a unit contrast vector at the specified angle
    targetDKLDir = [0 cos(theAngles(aa)) sin(theAngles(aa))]';

    % Convert from DKL to cone contrast to cone excitation direction.
    % Don't care about length here as that is handled by the contrast
    % maximization code below.
    theLMSExcitations = ContrastToExcitation(M_ConeIncToConeContrast*...
        M_DKLToConeInc*targetDKLDir,bgLMS);

    % Convert the direction to the desired direction in primary space.
    % Since this is desired, we do not go into settings here. Adding
    % and subtracting the background handles the ambient correctly.
    thePrimaryDir = SensorToPrimary(calObjCones,theLMSExcitations) - ...
        SensorToPrimary(calObjCones,bgLMS);

    % Find out how far we can go in the desired direction and scale the
    % unitPrimaryDir by that amount.
    % 
    % Using s rather than sPos here seems a little conservative, but when
    % we try sPos we find we are out of gamut in some cases.  
    [s,sPos,sNeg] = MaximizeGamutContrast(thePrimaryDir,bgPrimary);
    gamutPrimaryDir = sPos*thePrimaryDir;
    if (any(gamutPrimaryDir+bgPrimary < -1e-3) || any(gamutPrimaryDir+bgPrimary > 1+1e-3))
        error('Somehow primaries got too far out of gamut\n');
    end
    gamutDevPos1 = abs(gamutPrimaryDir+bgPrimary - 1);
    gamutDevNeg1 = abs(gamutPrimaryDir+bgPrimary);
    gamutDevPos2 = abs(-gamutPrimaryDir+bgPrimary - 1);
    gamutDevNeg2 = abs(-gamutPrimaryDir+bgPrimary);
    gamutDev = min([gamutDevPos1 gamutDevNeg1 gamutDevPos2 gamutDevNeg2]);
    if (gamutDev > 1e-3)
        error('Did not get primaries close enough to gamut edge');
    end

    % Get the settings that as closely as possible approximate what we
    % want.  One of these should be very close to 1 or 0, and none should
    % be less than 0 or more than 1.
    gamut_bg_primary(:,aa) = gamutPrimaryDir + bgPrimary;
    [gamutSettings(:,aa),badIndex] = PrimaryToSettings(calObjCones,...
        gamut_bg_primary(:,aa));
    if (any(badIndex))
        error('Somehow settings got out of gamut\n');
    end

    % Figure out the cone excitations for the settings we computed, and
    % then convert to contrast as our maximum contrast in this direction.
    gamutLMS(:,aa) = SettingsToSensor(calObjCones,gamutSettings(:,aa));
    gamutContrast(:,aa) = ExcitationToContrast(gamutLMS(:,aa),bgLMS);
    gamutDKL(:,aa) = M_ConeIncToDKL*M_ConeContrastToConeInc*gamutContrast(:,aa);
    vectorLengthGamutContrast(aa) = norm(gamutContrast(:,aa));
end
gamutDKLPlane = gamutDKL(2:3,:);

%% Make a plot of the gamut in the RGB space
fig_rgb = figure; ax_rgb = axes(fig_rgb);
%add walls
fill3(ax_rgb,[1,0,0,1],[0,0,0,0],[0,0,1,1],'k','FaceAlpha',0.05); hold on
fill3(ax_rgb,[0,0,0,0],[1,0,0,1],[0,0,1,1],'k','FaceAlpha',0.05); 
fill3(ax_rgb,[1,0,0,1],[1,1,1,1],[0,0,1,1],'k','FaceAlpha',0.05); 
fill3(ax_rgb,[1,1,1,1],[1,0,0,1],[0,0,1,1],'k','FaceAlpha',0.05); 
%real plots
f_rgb_1 = scatter3(ax_rgb,bgPrimary(1),bgPrimary(2),bgPrimary(3),200,'g+','lineWidth',2);  
f_rgb_2 = scatter3(ax_rgb,gamut_bg_primary(1,:), gamut_bg_primary(2,:), ...
    gamut_bg_primary(3,:), 5, 'k', 'filled'); 
    % gamut_bg_primary(3,:), 20, 'k', 'filled', 'MarkerEdgeColor','g'); 
fill3(ax_rgb, gamut_bg_primary(1,:), gamut_bg_primary(2,:), ...
    gamut_bg_primary(3,:), 'k','FaceColor','k','FaceAlpha',0.3);
for aa = 1:25:nAngles
    vec_aa = horzcat(bgPrimary, gamut_bg_primary(:,aa));
    plot3(ax_rgb, vec_aa(1,:), vec_aa(2,:), vec_aa(3,:),'k-.');
end
xlim([0,1]); ylim([0,1]); zlim([0,1]); xlabel('R'); ylabel('G'); zlabel('B')
axis square; grid on

%%  Make a plot of the gamut in the DKL isoluminant plane
fig_dkl = figure; ax_dkl = axes(fig_dkl);
[X, Y, Z] = sphere; n_contour = size(Z,2);
surf(ax_dkl, X, Y, Z,'FaceColor', 'k', 'FaceAlpha', 0.01,'EdgeColor',...
    [0.8,0.8,0.8]); hold on
f_dkl_1 = fill3(ax_dkl, X(ceil(n_contour/2),:),Y(ceil(n_contour/2),:),...
    Z(ceil(n_contour/2),:), 'k','FaceAlpha',0.1,'EdgeColor','none'); 
f_dkl_2 = fill3(ax_dkl, gamutDKLPlane(1,:),gamutDKLPlane(2,:),zeros(nAngles),...
    'k','FaceColor','k', 'FaceAlpha',0.4);
xticks(-1:1:1); yticks(-1:1:1);zticks(-1:1:1);
xlabel('DKL L/(L+M)'); ylabel('DKL S'); zlabel('DKL lum');
axis square; grid on; view(-30,30);

%% Let's try to find the corners
%
% First define the corners that we think could be there,
% by defining the line segments a plane could intersect.
theGamutLineSegmentsPrimary = {[ [0 0 0]' [1 0 0]' ] ...
                               [ [0 0 0]' [0 1 0]' ] ...
                               [ [0 0 0]' [0 0 1]' ] ...
                               [ [1 0 0]' [1 1 0]' ] ...
                               [ [1 0 0]' [1 0 1]' ] ...
                               [ [0 1 0]' [1 1 0]' ] ...
                               [ [0 1 0]' [0 1 1]' ] ...
                               [ [0 0 1]' [1 0 1]' ] ...
                               [ [0 0 1]' [0 1 1]' ] ...
                               [ [1 1 0]' [1 1 1]' ] ...
                               [ [0 1 1]' [1 1 1]' ] ...
                               [ [1 0 1]' [1 1 1]' ] ...
                               };
numLineSeg = length(theGamutLineSegmentsPrimary); 

%initialize
[theGamutLineSegmentsLMS, theGamutLineSegmentsContrast, theGamutLineSegmentsDKL] = ...
    deal(cell(1, length(theGamutLineSegmentsPrimary)));
% Convert each of these into DKL
for ll = 1:numLineSeg
    theGamutLineSegmentsLMS{ll} = PrimaryToSensor(calObjCones,theGamutLineSegmentsPrimary{ll});
    theGamutLineSegmentsContrast{ll} = ExcitationToContrast(theGamutLineSegmentsLMS{ll},bgLMS);
    theGamutLineSegmentsDKL{ll} = M_ConeIncToDKL*M_ConeContrastToConeInc*theGamutLineSegmentsContrast{ll};
end

% Define the plane we'll intersect with these line segments
bgDKL = [0 0 0]';
planeBasisDKL = [ [0 1 0]' [0 0 1]' ];

% Find each intersection point of plane with the line.  This
% involves solving a set of linear equations for each line segment
% and checking whether the intersection is between the two
% endpoints.
%initialize
[lineSegmentFactor, corner] = deal(NaN(1, numLineSeg));
[intersectingPoints, intersectingPoints1] = deal(NaN(3, numLineSeg));
for ll = 1:numLineSeg
    lineSegmentBase = theGamutLineSegmentsDKL{ll}(:,1);
    lineSegmentDelta = theGamutLineSegmentsDKL{ll}(:,2) - theGamutLineSegmentsDKL{ll}(:,1);
    lhs = [planeBasisDKL -lineSegmentDelta];
    rhs =  lineSegmentBase - bgDKL;
    intersectionVector = lhs\rhs;
    lineSegmentFactor(ll) = intersectionVector(3);

    % These two ways of computing the intersecting point should agree as
    % that is the equation we solved
    intersectingPoints(:,ll) = planeBasisDKL*intersectionVector(1:2) + bgDKL;
    intersectingPoints1(:,ll) = lineSegmentBase + lineSegmentFactor(ll)*lineSegmentDelta;
    if (lineSegmentFactor(ll) >= 0 && lineSegmentFactor(ll) <= 1)
        corner(ll) = true;
        f_dkl_3 = plot3(ax_dkl, intersectingPoints(2,ll),intersectingPoints(3,ll), bgDKL(1),...
            'bo','MarkerFaceColor','b','MarkerSize',14);
        f_dkl_4 = plot3(ax_dkl, intersectingPoints1(2,ll),intersectingPoints1(3,ll),bgDKL(1),...
            'ro','MarkerFaceColor','r','MarkerSize',10);
    else
        corner(ll) = false;
    end
end

%% Select out the corner coordinates in 2D
%there are two ways of finding the corner points in the RGB space
%1: find the angle indices that correspond to the corners
cornerIndices = find(corner);
numCorners = length(cornerIndices);
cornerPointsDKLPlane = intersectingPoints1(2:3,cornerIndices);
DKL_diff_val = arrayfun(@(idx) sum(abs(cornerPointsDKLPlane(:,idx) - gamutDKL(2:3,:))),...
    1:length(cornerIndices), 'UniformOutput', false);
angleIndices = arrayfun(@(idx) find(DKL_diff_val{idx} == min(DKL_diff_val{idx}), 1, 'first'),...
    1:length(cornerIndices));
%retrieve the corner points in the RGB space
cornerPointsRGB = gamut_bg_primary(:, angleIndices);
f_rgb_3 = plot3(ax_rgb, cornerPointsRGB(1,:), cornerPointsRGB(2,:),...
    cornerPointsRGB(3,:), 'bo','MarkerFaceColor','b','MarkerSize',14); 

%2: go through all the color spaces from DKL to RGB
corner_DKLDir = vertcat(zeros(1,numCorners),cornerPointsDKLPlane)';
corner_theLMSExcitations_temp = arrayfun(@(idx) ContrastToExcitation(M_ConeIncToConeContrast*...
    M_DKLToConeInc*corner_DKLDir(idx,:)',bgLMS), 1:numCorners,'UniformOutput', false); %loop through the four corners
corner_theLMSExcitations = cell2mat(corner_theLMSExcitations_temp);
% we the LMS cone excitations, we can them multiply the transformation
% matrix M_LMSTORGB. Note that we have to subtract the amblient LMS here
corner_PointsRGB = M_LMSTORGB*(corner_theLMSExcitations - ambientLMS);
f_rgb_4 = plot3(ax_rgb, corner_PointsRGB(1,:), corner_PointsRGB(2,:),...
    corner_PointsRGB(3,:), 'ro','MarkerFaceColor','r','MarkerSize',10); 

%% transformations trom DKL to W space, and RGB space
bgDKLPlane = [bgDKL(2:3)]; %1st dim: lum; 2nd dim: L/(L+M); 3rd dim: S
numCor = length(cornerIndices);
use_builtInTrans = true;

% If there are 4 corners, try to map to W space.
if (numCorners == 4)
    %target corners have to be in the same order as the source corners
    %the order is: bottom left, bottom right, top left, top right
    sourceCorners = cornerPointsDKLPlane'-bgDKLPlane';
    targetCorners = [ [-1 -1]' [1 -1]' [-1 1]' [1 1]' ];

    if use_builtInTrans
        %use matlab built-in function to solve for the transformation matrix
        tform_DKLPlaneTo2DW = fitgeotform2d(sourceCorners, targetCorners',"projective");
        M_DKLPlaneTo2DW = tform_DKLPlaneTo2DW.A;
    else
        % resource: https://cseweb.ucsd.edu/classes/wi07/cse252a/homography_estimation/homography_estimation.pdf
        % Initialize matrix M
        M = [];
        % Populate A with the equations
        for i = 1:numCor
            xi = cornerPointsDKLPlane(1,i);
            yi = cornerPointsDKLPlane(2,i);
            x_prime = targetCorners(1,i);
            y_prime = targetCorners(2,i);
            
            M = [M;
                 -xi, -yi, -1, 0, 0, 0, xi*x_prime, yi*x_prime, x_prime;
                 0, 0, 0, -xi, -yi, -1, xi*y_prime, yi*y_prime, y_prime];
            % Perform SVD to solve for h
            [~, ~, V] = svd(M);
            % The last column of V gives the solution
            h = V(:, end);
            M_DKLPlaneTo2DW = reshape(h, 3, 3)';
        end
    end
    %the transformation matrix is 3 x 3, but the source corners have size
    %of 2 x 4, so we need to adding a third column of ones
    cornerPoints2DW_temp = M_DKLPlaneTo2DW*vertcat(sourceCorners', ones(1,numCor));
    %Divide the first two columns by the third column to get the final Nx2 transformed points.
    cornerPoints2DW = cornerPoints2DW_temp(1:2,:) ./ cornerPoints2DW_temp(3,:);

    %repeat the same for all the points along the monitor's gamut
    gamut2DW_temp = M_DKLPlaneTo2DW*vertcat(gamutDKLPlane-bgDKLPlane,ones(1,nAngles));
    gamut2DW = gamut2DW_temp(1:2,:)./gamut2DW_temp(3,:);
    
    % Make a plot of the gamut in the DKL isoluminantplane
    fig_W = figure; ax_W = axes(fig_W); hold on;
    f_W_1 = plot(ax_W, gamut2DW(1,:),gamut2DW(2,:),'k','LineWidth',2);
    f_W_2 = plot(ax_W, targetCorners(1,:), targetCorners(2,:),'bo','MarkerFaceColor','b','MarkerSize',14);
    f_W_3 = plot(ax_W, cornerPoints2DW(1,:),cornerPoints2DW(2,:),'ro','MarkerFaceColor','r','MarkerSize',10);
    xlim([-1.2 1.2]);  ylim([-1.2 1.2]); axis('square');
    xlabel('Wishart space dim 1'); ylabel('Wishart space dim 2');

    %% select 9 reference locations
    num_grid_pts = 5;
    ref_W_1d = linspace(-0.6,0.6,num_grid_pts);
    [ref_W_x, ref_W_y] = meshgrid(ref_W_1d, ref_W_1d);
    nRef = length(ref_W_x(:));
    
    %convert it back to DKL space
    ref_W = [ref_W_x(:), ref_W_y(:)];
    ref_W_ext = [ref_W, ones(nRef,1)]';
    %compute the inverse of the transformation matrix so that it takes us from DKL to W
    M_2DWToDLKPlane = inv(M_DKLPlaneTo2DW);
    ref_dkl = M_2DWToDLKPlane*ref_W_ext;
    %Divide the first two columns by the third column to get the final Nx2 transformed points.
    %also add zeros to the last row because we 
    ref_dkl_norm = ref_dkl(1:2,:)./ref_dkl(3,:);
    ref_dkl_ext = vertcat(bgDKL(1).*ones(1,nRef),ref_dkl_norm);
    %convert it back to RGB space
    ref_theLMSExcitations = arrayfun(@(idx) ContrastToExcitation(M_ConeIncToConeContrast*...
        M_DKLToConeInc*ref_dkl_ext(:,idx),bgLMS), 1:nRef,'UniformOutput', false); %loop through the four corners
    ref_rgb = M_LMSTORGB*(cell2mat(ref_theLMSExcitations) - ambientLMS);
    
    %% add ref to the plots
    cmap = colormap('parula');
    colors_W = cmap(round(linspace(1, size(cmap, 1), nRef)), :);
    
    scatter(ax_W, ref_W_x(:), ref_W_y(:),100, colors_W,...
        'filled','MarkerEdgeColor','k','Marker','o','lineWidth',2);
    % legend(ax_W, [f_W_1, f_W_2, f_W_3], ...
    %     {'The monitor''s gamut', ...
    %     'Target corners (pre-specified)',...
    %     sprintf('Target corners (multiplying transformation matrix\n and source corners in DKL space)')},...
    %     'Location','north');
    lgd_W = legend(ax_W, [f_W_1, f_W_2], ...
        {'The monitor''s gamut', ...
        'Corners'},...
        'Location','northoutside', 'Orientation', 'vertical');
    set(ax_W, 'XTick', -0.6:0.3:0.6);
    set(ax_W, 'YTick', -0.6:0.3:0.6); grid on; box on;
    set(ax_W,'FontSize',12);
    set(fig_W, 'PaperSize', [10, 10]);
    pdf_filename1 = fullfile([cal_path, '/Plots'], sprintf('W_space_%s.pdf',whichCalFile(1:end-4)));
    saveas(fig_W, pdf_filename1);
    %print(fig_W, pdf_filename1, '-dpdf', '-opengl', '-bestfit');
    
    scatter3(ax_dkl, ref_dkl_norm(1,:), ref_dkl_norm(2,:), bgDKL(1).*ones(1, nRef),100,colors_W,...
        'filled', 'MarkerEdgeColor','k','Marker','o','lineWidth',2);
    % legend(ax_dkl,[f_dkl_1, f_dkl_2(1), f_dkl_3, f_dkl_4], ...
    %     {'Isoluminant plane in DKL space', ...
    %     'The monitor''s gamut',...
    %     'Corner points computed using method 1',...
    %     'Corner points computed using method 2'},...
    %     'Location','north');
    lgd_dkl = legend(ax_dkl,[f_dkl_1, f_dkl_2(1), f_dkl_3], ...
        {'Isoluminant plane in DKL space', ...
        'The monitor''s gamut',...
        'Corners'},...
        'Location','northoutside', 'Orientation', 'vertical');
    set(ax_dkl,'FontSize',12);
    set(fig_dkl, 'PaperSize', [10, 10]);
    pdf_filename2 = fullfile([cal_path, '/Plots'], sprintf('dkl_space_%s.pdf',whichCalFile(1:end-4)));
    %print(fig_dkl, pdf_filename2, '-dpdf', '-opengl', '-bestfit');
    saveas(fig_dkl, pdf_filename2);
    
    scatter3(ax_rgb, ref_rgb(1,:), ref_rgb(2,:), ref_rgb(3,:),100,colors_W,...
        'filled', 'MarkerEdgeColor','k','Marker','o','lineWidth',2);
    % legend(ax_rgb,[f_rgb_1, f_rgb_2, f_rgb_3, f_rgb_4], ...
    %     {'Background primary',...
    %     'The monitor''s gamut',...
    %     'Corner points computed using angle indices corresponding to corner points',...
    %     'Corner points computed using the transformation matrix from LMS to RGB'},...
    %     'Location','north');
    lgd_rgb = legend(ax_rgb,[f_rgb_1, f_rgb_2, f_rgb_3], ...
        {'Background primary',...
        'The monitor''s gamut',...
        'Corners'},...
        'Location','northoutside', 'Orientation', 'vertical');
    set(ax_rgb, 'XTick', 0.2:0.15:0.8);
    set(ax_rgb, 'YTick', 0.2:0.15:0.8); 
    set(ax_rgb,'FontSize',12);
    set(fig_rgb, 'PaperSize', [10, 10]);
    % Try forcing MATLAB to save as a vector PDF
    pdf_filename3 = fullfile([cal_path, '/Plots'], sprintf('rgb_space_%s.pdf',whichCalFile(1:end-4)));
    %print(fig_rgb, pdf_filename3, '-dpdf', '-opengl', '-bestfit');
end

%% sanity checks
% if the RGB dots are really from the isoluminant plane, then we should get
% the same luminance value by doing the calculation below
lum_check = NaN(1,size(ref_rgb, 2));
for i = 1:size(ref_rgb, 2)
    lum_check(i) = T_Y * (calObjXYZ.cal.P_device * ref_rgb(:,i));
end
assert(min(abs(lum_check - mean(lum_check))) < 1e-6,...
    'The dots are not on an isoluminant plane!')

%compute a matrix that converts W to RGB space without going through
%all the details above
M_2DWToRGB = ref_rgb * pinv(ref_W_ext);
M_RGBTo2DW = inv(M_2DWToRGB);
ref_rgb_check = M_2DWToRGB * ref_W_ext;
ref_W_ext_check = M_RGBTo2DW * ref_rgb;
assert(min(min(abs(ref_rgb_check - ref_rgb))) < 1e-6,...
    'The matrix that is supposed to take us from W to RGB is not computed correctly!')
assert(min(min(abs(ref_W_ext_check - ref_W_ext))) < 1e-6,...
    'The matrix that is supposed to take us from W to RGB is not computed correctly!')

scatter3(ax_rgb,ref_rgb_check(1,:), ref_rgb_check(2,:), ref_rgb_check(3,:), 'k+')
saveas(fig_rgb, pdf_filename3);

%% save almost all the variables for housekeeping purpose
% Get a list of all variables in the current workspace
vars = who;
% Initialize the struct with the specified name
eval([whichCalFile(1:end-4), ' = struct();'])
struct_temp = struct();
% Open a MAT-file to save the variables
matfileName = 'Transformation_btw_color_spaces.mat';
outputName = fullfile(cal_path,matfileName);
% Check if the file already exists
fileExists = isfile(outputName);

for i = 1:length(vars)
    % Get the variable name
    varName = vars{i};
    % Skip the following variables
    if ismember(varName, {'fig_dkl', 'fig_rgb','fig_W',...
                          'ax_dkl', 'ax_rgb','ax_W',...
                          'f_dkl_1','f_dkl_2','f_dkl_3','f_dkl_4',...
                          'f_rgb_1','f_rgb_2','f_rgb_3','f_rgb_4',...
                          'f_W_1','f_W_2','f_W_3',...
                          'ldg_dkl','lgd_rgb', 'lgd_W'})
        continue;
    end
    % Use dynamic field referencing to add the variable to the struct
    struct_temp.(varName) = eval(varName);
end
eval([whichCalFile(1:end-4), ' = struct_temp;']);

% Save or append the struct to the MAT file
if fileExists
    % Append if file already exists
    save(outputName, whichCalFile(1:end-4), '-append');
else
    % Save as new file if it does not exist
    save(outputName, whichCalFile(1:end-4));
end

%% save the two transformation matrix to .xlsx files
% Specify the filename
filename1 = 'M_2DWToRGB.xlsx';
outputName1 = fullfile(cal_path,filename1);
filename2 = 'M_RGBTo2DW.xlsx';
outputName2 = fullfile(cal_path,filename2);
% Save the matrix to a CSV file
writematrix(M_2DWToRGB, outputName1, 'Sheet', whichCalFile(1:13));
writematrix(M_RGBTo2DW, outputName2, 'Sheet', whichCalFile(1:13));


