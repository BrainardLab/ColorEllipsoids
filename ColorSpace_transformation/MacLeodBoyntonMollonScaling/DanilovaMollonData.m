%% DanilovaMollonData
%
% This script develops the data representation used by Danilova and
% Mollon and illustrates how we think we should connect their 
% data representation to ours.
%
% The connection is only approximate, because they used JuddVos to define
% the isoluminant plane, and we used CIE.  Also, there is a shift in the
% adapting chromaticity. 

% Initialize
clear; close all;

% Load spectral data and set calibration file
S = [390 1 441];
wls = SToWls(S);

% Set output directory for figures and stuff
outputDir = fullfile(pwd,'Output');
if (~exist(outputDir,'dir'))
    mkdir(outputDir);
end

% Get Danilova and Mollen fundamentals and associated luminance.
% We want to scale the LMS cones so that they sum to the specified
% luminance, to keep everything very happy.
load T_cones_sp
load T_xyzJuddVos
T_cones_dm = SplineCmf(S_cones_sp,T_cones_sp,S);
T_XYZ_dm = SplineCmf(S_xyzJuddVos,T_xyzJuddVos,S);
T_Y_dm = T_XYZ_dm(2,:);
[~, factorsLMS_dm] = LMSToMacBoyn([],T_cones_dm,T_Y_dm,1);
T_cones_dm = diag(factorsLMS_dm)*T_cones_dm;
M_LMSToXYZ_dm = (T_cones_dm'\T_XYZ_dm')';
if (sum(abs(T_cones_dm(1,:) + T_cones_dm(2,:) - T_Y_dm)) > 1e-10)
    error('Oops');
end

% Get our fundamentals and associated luminance.
load T_cones_ss2
load T_xyzCIEPhys2
T_cones_us = SplineCmf(S_cones_ss2,T_cones_ss2,S);
T_XYZ_us = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
T_Y_us = T_XYZ_us(2,:);
[~, factorsLMS_us] = LMSToMacBoyn([],T_cones_us,T_Y_us,1);
T_cones_us = diag(factorsLMS_us)*T_cones_us;
M_LMSToXYZ_us = (T_cones_us'\T_XYZ_us')';
if (sum(abs(T_cones_us(1,:) + T_cones_us(2,:) - T_Y_us)) > 1e-4)
    error('Oops');
end

% Get spectrum of D65
load spd_D65.mat
spd_D65 = SplineSpd(S_D65,spd_D65,S);
LMSD65_dm = T_cones_dm*spd_D65;
lumD65_dm = T_Y_dm*spd_D65;

% Recreate the spectrum locus and equal energy white
lsYSpectrumLocus_dm = LMSToMacBoyn(T_cones_dm,T_cones_dm,T_Y_dm,1);
xyYSpectrumLocus_dm = XYZToxyY(T_XYZ_dm);
index574 = find(wls == 574);
plotWls = SToWls([400 10 31]);
for ii = 1:length(plotWls)
    plotIndex(ii) = find(wls == plotWls(ii));
end

% Get point colors for spectrum locus using SRGB
plotColors = SRGBGammaCorrect(XYZToSRGBPrimary((T_XYZ_dm(:,plotIndex))));

% Compute representations for equal energy white.
% Scale by hand to make the plotted point end up
% in a reasonable place in the 3D plot.
EEFactor = 200;
LMSEEWhite_dm = sum(T_cones_dm,2)/EEFactor;
XYZEEWhite_dm = M_LMSToXYZ_dm*LMSEEWhite_dm;
xyYEEWhite_dm = XYZToxyY(XYZEEWhite_dm);
lsYEEWhite_dm = LMSToMacBoyn(LMSEEWhite_dm,T_cones_dm,T_Y_dm,1);
LMSEEWhiteCheck_dm = MacBoynToLMS(lsYEEWhite_dm, T_cones_dm, T_Y_dm);
if (max(abs(LMSEEWhiteCheck_dm - LMSEEWhite_dm)) > 1e-4)
    error('Inversion failure');
end

% Compute representations for D65
lsYD65_dm = LMSToMacBoyn(LMSD65_dm,T_cones_dm,T_Y_dm,1);
LMSD65Check_dm = MacBoynToLMS(lsYD65_dm, T_cones_dm, T_Y_dm);
if (max(abs(LMSD65Check_dm - LMSD65_dm)) > 1e-4)
    error('Inversion failure');
end

% Find scale factor for S cone axis that makes line between 574 and D65 %
% run at 45 deg.  This is how Mollon and Danilova scale their MB space
% in many of their papers.  This method should work for any choice of
% underlying cones.
lsY574_dm = lsYSpectrumLocus_dm(:,index574);
deltaD65574_dm = lsY574_dm(1:2)-lsYD65_dm(1:2);
desiredS_dms = deltaD65574_dm(1) + lsY574_dm(2); 
dmSFactor = desiredS_dms/lsYD65_dm(2);

% Apply Danilova/Mollon scaling for MB space?
dmScale = true;
if (dmScale)
    lsYScaleMatrix_dm = diag([1 dmSFactor 1]);
else
    lsYScaleMatrix_dm = eye(2);
end
lsYScaleMatrixInv_dm = inv(lsYScaleMatrix_dm);

%% Convert into DM scaled Macleod-Boyton
lsYSpectrumLocus_dms = lsYScaleMatrix_dm*lsYSpectrumLocus_dm;
lsYEEWhite_dms = lsYScaleMatrix_dm*lsYEEWhite_dm;
lsYD65_dms = lsYScaleMatrix_dm*lsYD65_dm;
lsY574_dms = lsYScaleMatrix_dm*lsY574_dm;
deltalsYD65574_dms = lsY574_dms-lsYD65_dms;

% Compute end point of line through 574 and D65 at y axis
% in the DM scaled MB space
lMinValue = 0.59;
sIntValue = (lMinValue-lsY574_dm(1))*deltalsYD65574_dms(2)/deltalsYD65574_dms(1)+lsY574_dm(2);

%% Plot M-B diagram in DM scale
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot(lsYSpectrumLocus_dms(1,plotIndex(ii)),lsYSpectrumLocus_dms(2,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot(lsYSpectrumLocus_dms(1,:),lsYSpectrumLocus_dms(2,:),'k','LineWidth',2);
plot(lsYSpectrumLocus_dms(1,index574),lsYSpectrumLocus_dms(2,index574), ...
    'o','Color',[1 0 0],'MarkerFaceColor',[1 0 0],'MarkerSize',12);
scatter(lsYEEWhite_dms(1),lsYEEWhite_dms(2), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter(lsYD65_dms(1),lsYD65_dms(2), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
xlabel('l'); ylabel('s');
title({'MacLeod-Boynton'}, ...
    'FontName','Helvetica','FontSize',20);
xlim([0.4 1]); ylim([0,1]);
axis('square'); grid on
if dmScale; str_ext = '_wMollonScale'; else; str_ext = '';end
%saveas(gcf,fullfile(outputDir, sprintf('MacLeodBoynton%s.pdf', str_ext)),'pdf');

%% Zoomed version, as is typical in some papers. Compare
% with Danilova and Mollon 2012 Figure 1B.
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot(lsYSpectrumLocus_dms(1,plotIndex(ii)),lsYSpectrumLocus_dms(2,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot(lsYSpectrumLocus_dms(1,:),lsYSpectrumLocus_dms(2,:),'k','LineWidth',2);
plot(lsYSpectrumLocus_dms(1,index574),lsYSpectrumLocus_dms(2,index574), ...
    'o','Color',[1 0 0],'MarkerFaceColor',[1 0 0],'MarkerSize',12);
scatter(lsYEEWhite_dms(1),lsYEEWhite_dms(2), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter(lsYD65_dms(1),lsYD65_dms(2), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
plot([lsY574_dms(1) lMinValue],[lsY574_dms(2) sIntValue],'k:','LineWidth',2);
xlabel('l'); ylabel('s');
title({'MacLeod-Boynton'}, ...
    'FontName','Helvetica','FontSize',20);
axis('square');
xlim([0.59 0.7]); ylim([-0.01 0.1]);
set(gca,'YTick',[0 0.02 0.04 0.06 0.08 1.0]);
% saveas(gcf,fullfile(outputDir,'MacLeodBoyntonZoom.tiff'),'tif');

%% Get info to extract data from Danilova and Mollon 2025
%
% Generate unit circle
ell_pts = 1000;
t = linspace(0, 2*pi, ell_pts);
circle = [cos(t); sin(t)];  % 2xN matrix

%  Read in the ellipse table
ellipseDataTable = readtable('DanilovaMollon2025_Table1.xlsx','VariableNamingRule','preserve');
stimSizesMin = [32, 48, 240]; %in min

% Figure out both coordinates from table.  The table gives the l coordinate
% of the reference, and tells us the angle of the line in the space it lives in.
refLCoords = ellipseDataTable.Ref_LminusM'; % Extract l coordinates from the table

% Loop through the table rows (one row per reference)
refSCoords = NaN(1, height(ellipseDataTable));
[ellipseAxis1, ellipseAxis2, ellipseAngle] = deal(NaN(height(ellipseDataTable), length(stimSizesMin)));
[rotMat, stretchMat, ellMat] = deal(NaN(height(ellipseDataTable), length(stimSizesMin), 2, 2));
ellContour = NaN(height(ellipseDataTable), length(stimSizesMin), 2, ell_pts);
for rr = 1:height(ellipseDataTable)
    % If both l and s are there, just read them out, otherwise compute.
    if (ellipseDataTable.Ref_S(rr) == -1)
        % Need to figure out reference s coord
        deltaL = refLCoords(rr) - lsYD65_dms(1);
        deltaS = deltaL*tand(ellipseDataTable.LineAngle(rr));
        refSCoords(rr) = lsYD65_dms(2) + deltaS;
    else
        % We have reference s coord already
        refSCoords(rr) = ellipseDataTable.Ref_S(rr);
    end

    for ll = 1:length(stimSizesMin)
        eval(sprintf('ellipseAxis1(rr,ll) = ellipseDataTable.L1_%d(rr);', stimSizesMin(ll)));
        eval(sprintf('ellipseAxis2(rr,ll) = ellipseDataTable.L2_%d(rr);',  stimSizesMin(ll)));
        eval(sprintf('ellipseAngle(rr,ll) = ellipseDataTable.Theta_%d(rr);',  stimSizesMin(ll)));

        rotMat(rr,ll,:,:) = [cosd(ellipseAngle(rr,ll)), -sind(ellipseAngle(rr,ll)); ...
                             sind(ellipseAngle(rr,ll)), cosd(ellipseAngle(rr,ll))];
        stretchMat(rr,ll,:,:) = [ellipseAxis1(rr,ll)^2, 0;...
                              0, ellipseAxis2(rr,ll)^2];
        ellMat(rr,ll,:,:) = squeeze(rotMat(rr,ll,:,:)) * squeeze(stretchMat(rr,ll,:,:)) * squeeze(rotMat(rr,ll,:,:))';
    
        % Transform unit circle into ellipse using Cholesky or sqrtm
        ellContour(rr,ll,:,:) = sqrtm(squeeze(ellMat(rr,ll,:,:))) * circle;
    end
end

%% Plot the ellipses
scaler_vis = [1,1,4];
y_bds = [0, 0.2];
x_bds = [0.58, 0.78];

% Loop and plot ellipses for each stim size in its own subplot
figure; clf; hold on;
for ll = 1:length(stimSizesMin)
    subplot(1,length(stimSizesMin), ll);

    % Plot reference lines at different orientations
    plot(x_bds, ones(1,2).*lsYD65_dms(2), 'k:'); hold on
    plot(lsYD65_dms(1).*ones(1,2), y_bds, 'k:');
    plot(lsYD65_dms(1) + [1, -1], lsYD65_dms(2) + [tand(45),-tand(45)], 'k--');
    plot(lsYD65_dms(1) + [1, -1], lsYD65_dms(2) + [tand(135),-tand(135)], 'k--');
    plot(lsYD65_dms(1) + [1, -1], lsYD65_dms(2) + [tand(165), -tand(165)], 'k-');

    % Plot this ellpise
    for i = 1:height(ellipseDataTable)
        plot(refLCoords(i) + scaler_vis(ll).*squeeze(ellContour(i,ll,1,:)),...
            refSCoords(i) + scaler_vis(ll).*squeeze(ellContour(i,ll,2,:)), 'k', 'LineWidth', 2);
    end
    axis square; hold off
    xlabel('L/(L+M)'); ylabel('S/(L+M)');
    xlim(x_bds); ylim(y_bds);
    xticks(0.6:0.05:0.75); yticks(0:0.05:0.2);
    title(sprintf('D = %d min', stimSizesMin(ll)));
end
set(gcf,'Unit','Normalized','Position',[0, 0, 0.7,0.3]);

%% Convert the ref to DKL
%
% Get calibration info as we need this to define RGB and
% the Wishart space (which depends on monitor gamut).
calPath = getpref('BrainardLabToolbox','CalDataFolder');
monInfoName = 'Transformation_btw_color_spaces.mat';
monData = load(fullfile(calPath,monInfoName));

% Obtain transformation matrix from RGB to Wishart.
% Switching name to WD2 here, but can't change 2DW
% in read matrix.
M_RGBToWD2_us = monData.DELL_02242025_texture_right.M_RGBTo2DW;

% Get the monitor's primary
P_device = monData.DELL_02242025_texture_right.cal.processedData.P_device;
S_device = monData.DELL_02242025_texture_right.cal.rawData.S; 
P_device = SplineSpd(S_device, P_device, S);

% Get transformation into and out of dm and our cones
M_RGBToLMS_dm = T_cones_dm*P_device;
M_LMSTORGB_dm = inv(M_RGBToLMS_dm);
M_RGBToLMS_us = T_cones_us*P_device;
M_LMSTORGB_us = inv(M_RGBToLMS_us);

% Get various adaptation gray points
monGrayRGB = [0.5, 0.5, 0.5]';

% Get DM adapt luminance and scale LMSD65 to same luminance
% to obtain DM adapt LMS
adaptLum_dm = T_Y_dm * (P_device * monGrayRGB);
adaptLMS_dm = LMSD65_dm*adaptLum_dm/lumD65_dm;

% Get our adaptation LMS and lumiannce from monitor gray point
adaptLMS_us = T_cones_us * (P_device * monGrayRGB);
adaptLum_us = T_Y_us * (P_device * monGrayRGB);

% DKL transformation matrices
M_ConeIncToDKL_dm = ComputeDKL_M(adaptLMS_dm,T_cones_dm,T_Y_dm);
M_DKLToConeInc_dm = inv(M_ConeIncToDKL_dm);
M_ConeIncToDKL_us = ComputeDKL_M(adaptLMS_us,T_cones_us,T_Y_us);
M_DKLToConeInc_us = inv(M_ConeIncToDKL_us);

% Ellipse references lsY_dm -> LMS_dm -> DKL -> LMS_us -> RGB_us -> WD2_us
%
% Also compute lsY_us, just for fun and because Y should be constant
% so it is a good check.
%
% Put in adapt luminance and DM unscale the Macleod-Boynton lsY coords.
% Note that adaptLum_dm = adaptLum_dms so we just carry the _dm version
% around.
lsYRefs_dm = lsYScaleMatrixInv_dm*[refLCoords ; refSCoords ; adaptLum_dm(1,ones(1,length(refSCoords)))];

% lsY_dm -> LMS_dm
LMSRef_dm = MacBoynToLMS(lsYRefs_dm, T_cones_dm, T_Y_dm);
lsYRefsCheck_dm = LMSToMacBoyn(LMSRef_dm, T_cones_dm, T_Y_dm, 1);
if (max(abs(lsYRefsCheck_dm(:) - lsYRefs_dm(:))) > 1e-4)
    error('We failed to recover the reference stimuli lsY from LMS.');
end

% LMS_dm -> DKL
LMSIncRef_dm = LMSRef_dm - adaptLMS_dm;
DKLRef = M_ConeIncToDKL_dm*LMSIncRef_dm;

% DKL -> LMS_us
LMSIncRef_us = M_DKLToConeInc_us*DKLRef;
LMSRef_us = LMSIncRef_us + adaptLMS_us;

% LMS_us -> RGB_us
RGBRef_us = M_LMSTORGB_us * LMSRef_us;
nOutOfGamutHigh = 0;
nOutOfGamutLow = 0;
for rr = 1:size(RGBRef_us,2)
    outOfGamutIndex(rr) = false;
    if (any(RGBRef_us(:,rr) > 1))
        nOutOfGamutHigh = nOutOfGamutHigh + 1;
        outOfGamutIndex(rr) = false;

    end
    if (any(RGBRef_us(:,rr) < 0))
        nOutOfGamutLow = nOutOfGamutLow + 1;
        outOfGamutIndex(rr) = false;
    end
end
fprintf('%d of %d refs out of gamut high\n',nOutOfGamutHigh,size(RGBRef_us,2));
fprintf('%d of %d refs out of gamut low\n',nOutOfGamutLow,size(RGBRef_us,2));

% Check luminance at this stage
lumRefCheck_us = T_Y_us*P_device*RGBRef_us;
if (max(abs(lumRefCheck_us - adaptLum_us)) > 1e-6)
    error('Luminance check failed');
end

% RGB_us -> WD2_us
WD2Ref_us = M_RGBToWD2_us * RGBRef_us;

% LMSRef_us -> lsYRef_us
lsYRef_us = LMSToMacBoyn(LMSRef_us, T_cones_us, T_Y_us, 1);

% If everything worked so far, the 3rd column of lsYRef_us should be 
% equal to adaptLum_us
if (max(abs(lsYRef_us(3,:) - adaptLum_us)) > 1e-6)
    error('Luminance check failed');
end

% Check that third coord of WD2Ref_us is all 1's.
if (max(abs(WD2Ref_us(3,:) - 1)) > 1e-4)
    error('Wishart space third coord error');
end




