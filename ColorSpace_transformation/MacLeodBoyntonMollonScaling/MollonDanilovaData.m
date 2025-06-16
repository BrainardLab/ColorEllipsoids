% Initialize
clear; close all; clc;

% Load spectral data and set calibration file
S = [390 1 441];
%whichCones = 'SmithPokornyJudd1951';
whichCones = 'SmithPokorny';
outputDir = fullfile(pwd,whichCones);
if (~exist(outputDir,'dir'))
    mkdir(outputDir);
end

switch (whichCones)
    case 'SmithPokorny'
		load T_cones_sp
		load T_xyzJuddVos
		T_cones = SplineCmf(S_cones_sp,T_cones_sp,S);
		T_XYZ = SplineCmf(S_xyzJuddVos,T_xyzJuddVos,S);
    case 'DemarcoPokornySmith'
		load T_cones_dps
		load T_xyzJuddVos
		T_cones = SplineCmf(S_cones_dps,T_cones_dps,S);
		T_XYZ = SplineCmf(S_xyzJuddVos,T_xyzJuddVos,S);
    case 'SmithPokornyJudd1951'
		load T_cones_sp
		load T_xyzJuddVos
        load T_Y_Judd1951;
        T_Y = SplineCmf(S_Y_Judd1951,T_Y_Judd1951,S);
		T_cones = SplineCmf(S_cones_sp,T_cones_sp,S);
		T_XYZ = SplineCmf(S_xyzJuddVos,T_xyzJuddVos,S);
        T_XYZ(2,:) = T_Y;
	case 'StockmanSharpe'
		load T_cones_ss2
		load T_xyzCIEPhys2
		T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);
        T_XYZ = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);
    otherwise
        error('Unknown cone fundamentals specified')
end

% Set up luminance and xform matrix
T_Y = T_XYZ(2,:);
M_LMSToXYZ = (T_cones'\T_XYZ')';

% Get spectrum of D65
load spd_D65.mat
spd_D65 = SplineSpd(S_D65,spd_D65,S);

% Check standard D65 chrom
load T_xyz1931
T_XYZ_1931 = SplineCmf(S_xyz1931,T_xyz1931,S);
xyY1931D65 = XYZToxyY(T_XYZ_1931*spd_D65);
fprintf('D65 chrom: %0.3f, %0.3f\n',xyY1931D65(1),xyY1931D65(2));

% Recreate the spectrum locus and equal energy white shown in Figure 8.2
% of CIE 170-2:2015 (if StockmanSharpe set). Otherwise make the
% MacLeod-Boynton diagram from Smith-Pokorny land, which differs from the
% CIE standard.
lsYSpectrumLocus = LMSToMacBoyn(T_cones,T_cones,T_Y,1);
xyYSpectrumLocus = XYZToxyY(T_XYZ);
wls = SToWls(S);
index574 = find(wls == 574);
plotWls = SToWls([400 10 31]);
for ii = 1:length(plotWls)
    plotIndex(ii) = find(wls == plotWls(ii));
end

% Get point colors for spectrum locus using SRGB
plotColors = SRGBGammaCorrect(XYZToSRGBPrimary((T_XYZ(:,plotIndex))));

% Compute the sum of the ls values in the spectrum locus, and compare
% to the value that this example computed in February 2019, entered
% here to four places as 412.2608.  This comparison provides a
% check that this routine still works the way it did when we put in the
% check.
if (strcmp(whichCones,'StockmanSharpe'))
    temp = lsYSpectrumLocus(1:2,:);
    check = round(sum(temp(:)),4);
    if (abs(check-412.2608) > 1e-4)
        error('No longer get same check value as we used to');
    end
end

% Compute representations for equal energy white.
% Scale by hand to make the plotted point end up
% in a reasonable place in the 3D plot.
EEFactor = 200;
LMSEEWhite = sum(T_cones,2)/EEFactor;
XYZEEWhite = M_LMSToXYZ*LMSEEWhite;
xyYEEWhite = XYZToxyY(XYZEEWhite);
lsYEEWhite = LMSToMacBoyn(LMSEEWhite,T_cones,T_Y,1);
%FH: to verify MacBoynToLMS
recover_LMSEEWhite = MacBoynToLMS(lsYEEWhite, T_cones, T_Y);
fprintf('Difference (Recovered - Original):\n');
disp(round(recover_LMSEEWhite - LMSEEWhite,4));

% Compute representations for D65
D65Factor = 4000;
LMSD65 = T_cones*spd_D65/D65Factor;
XYZD65 = M_LMSToXYZ*LMSD65;
xyYD65 = XYZToxyY(XYZD65);
lsYD65 = LMSToMacBoyn(LMSD65,T_cones,T_Y,1);
%FH: to verify MacBoynToLMS
recover_LMSD65 = MacBoynToLMS(lsYD65, T_cones, T_Y);
fprintf('Difference (Recovered - Original):\n');
disp(round(recover_LMSD65 - LMSD65,4));

% Find scale factor for S cone axis that makes line between 574 and D65 %
% run at 45 deg.  This is how Mollon and Danilova scale their MB space
% in many of their papers.  This method should work for any choice of
% underlying cones.
ls574 = lsYSpectrumLocus(1:2,index574);
deltaD65574 = ls574-lsYD65(1:2);
desiredS = deltaD65574(1) + ls574(2); 
mollonFactor = desiredS/lsYD65(2);

% Scale to Mollon/Danilova land if desired
% Apply Mollon/Danilova scaling for MB space?
mollonScale = true;
if (mollonScale)
    mollonScaleMatrix = diag([1 mollonFactor]);
else
    mollonScaleMatrix = eye(2);
end
lsYSpectrumLocus(1:2,:) = mollonScaleMatrix*lsYSpectrumLocus(1:2,:);
lsYEEWhite(1:2) = mollonScaleMatrix*lsYEEWhite(1:2);
lsYD65(1:2) = mollonScaleMatrix*lsYD65(1:2);
ls574 = mollonScaleMatrix*ls574;
deltaD65574 = ls574-lsYD65(1:2);

% Compute end point of line through 574 and D65 at y axis
lMinValue = 0.59;
sIntValue = (lMinValue-ls574(1))*deltaD65574(2)/deltaD65574(1)+ls574(2);

%% Plot M-B diagram
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot(lsYSpectrumLocus(1,plotIndex(ii)),lsYSpectrumLocus(2,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot(lsYSpectrumLocus(1,:),lsYSpectrumLocus(2,:),'k','LineWidth',2);
plot(lsYSpectrumLocus(1,index574),lsYSpectrumLocus(2,index574), ...
    'o','Color',[1 0 0],'MarkerFaceColor',[1 0 0],'MarkerSize',12);
scatter(lsYEEWhite(1),lsYEEWhite(2), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter(lsYD65(1),lsYD65(2), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
xlabel('l'); ylabel('s');
title({'MacLeod-Boynton' ; ['Based on ' whichCones]}, ...
    'FontName','Helvetica','FontSize',20);
xlim([0.4 1]); ylim([0,1]);
axis('square'); grid on
if mollonScale; str_ext = '_wMollonScale'; else; str_ext = '';end
%saveas(gcf,fullfile(outputDir, sprintf('MacLeodBoynton%s.pdf', str_ext)),'pdf');

%% Zoomed version, as is typical in some papers. Compare
% with Danilova and Mollon 2012 Figure 1B.
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot(lsYSpectrumLocus(1,plotIndex(ii)),lsYSpectrumLocus(2,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot(lsYSpectrumLocus(1,:),lsYSpectrumLocus(2,:),'k','LineWidth',2);
plot(lsYSpectrumLocus(1,index574),lsYSpectrumLocus(2,index574), ...
    'o','Color',[1 0 0],'MarkerFaceColor',[1 0 0],'MarkerSize',12);
scatter(lsYEEWhite(1),lsYEEWhite(2), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter(lsYD65(1),lsYD65(2), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
plot([ls574(1) lMinValue],[ls574(2) sIntValue],'k:','LineWidth',2);
xlabel('l'); ylabel('s');
title({'MacLeod-Boynton' ; ['Based on ' whichCones]}, ...
    'FontName','Helvetica','FontSize',20);
axis('square');
if (strcmp(whichCones,'StockmanSharpe'))
    xlim([0.59 0.7]); ylim([-0.01 0.1]);
    set(gca,'YTick',[0 0.02 0.04 0.06 0.08 1.0]);
elseif (strcmp(whichCones,'SmithPokorny') | ...
        strcmp(whichCones,'SmithPokornyJudd1951') | ...
        strcmp(whichCones,'DemarcoPokornySmith'))
    xlim([0.59 0.7]); ylim([-0.01 0.1]);
    set(gca,'YTick',[0 0.02 0.04 0.06 0.08 1.0]);
elseif (strcmp(whichCones,'SmithPokornyJudd1951'))
    xlim([0.59 0.7]); ylim([-0.01 0.1]);
    set(gca,'YTick',[0 0.02 0.04 0.06 0.08 1.0]);
end
% saveas(gcf,fullfile(outputDir,'MacLeodBoyntonZoom.tiff'),'tif');

% Plot xy chromaticity
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot(xyYSpectrumLocus(1,plotIndex(ii)),xyYSpectrumLocus(2,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot(xyYSpectrumLocus(1,:),xyYSpectrumLocus(2,:),'k','LineWidth',2);
scatter(xyYEEWhite(1),xyYEEWhite(2), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter(xyYEEWhite(1),xyYEEWhite(2), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
xlabel('x'); ylabel('y');
if (strcmp(whichCones,'StockmanSharpe'))
    title({'Chromaticity' ; 'CIE Physiological XYZ'},'FontName','Helvetica','FontSize',20);
elseif (strcmp(whichCones,'SmithPokorny'))
    title({'Chromaticity' ; 'Judd-Vos XYZ'},'FontName','Helvetica','FontSize',20);
end
xlim([0 1]); ylim([0 1]);
axis('square');
% saveas(gcf,fullfile(outputDir,'xyChrom.tiff'),'tif');

% Plot spectrum locus and EE white in LMS
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot3(T_cones(1,plotIndex(ii)),T_cones(2,plotIndex(ii)),T_cones(3,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot3(T_cones(1,:),T_cones(2,:),T_cones(3,:),'k','LineWidth',2);
scatter3(LMSEEWhite(1),LMSEEWhite(2),LMSEEWhite(3), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter3(LMSD65(1),LMSD65(2),LMSD65(3), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
xlabel('L'); ylabel('M'); zlabel('S');
title({'LMS Space' ; whichCones},'FontName','Helvetica','FontSize',20);
view([50 16]);
axis('square');
xlim([0 1]); ylim([0 1]); zlim([0 1]);
set(gca,'XTick',[0 0.5 1]);
set(gca,'YTick',[0 0.5 1]);
set(gca,'ZTick',[0 0.5 1]);
% saveas(gcf,fullfile(outputDir,'LMSSpace.tiff'),'tif');

% Plot spectrum locus and EE white in XYZ
figure; clf; hold on;
set(gca,'FontName','Helvetica','FontSize',16);
for ii = 1:length(plotIndex)
    plot3(T_XYZ(1,plotIndex(ii)),T_XYZ(2,plotIndex(ii)),T_XYZ(3,plotIndex(ii)), ...
        'o','Color',plotColors(:,ii)/255,'MarkerFaceColor',plotColors(:,ii)/255,'MarkerSize',12);
end
plot3(T_XYZ(1,:),T_XYZ(2,:),T_XYZ(3,:),'k','LineWidth',2);
scatter3(XYZEEWhite(1),XYZEEWhite(2),XYZEEWhite(3), 200,'Marker','s',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
scatter3(XYZD65(1),XYZD65(2),XYZD65(3), 200, 'Marker','d',...
    'MarkerFaceColor', [0.5 0.5 0.5],'MarkerFaceAlpha',0.5, 'MarkerEdgeColor','k');
xlabel('X');
ylabel('Y');
zlabel('Z');
if (strcmp(whichCones,'StockmanSharpe'))
    title({'XYZ Space' ; 'CIE Physiological XYZ'},'FontName','Helvetica','FontSize',20);
elseif (strcmp(whichCones,'SmithPokorny'))
    title({'XYZ Space' ; 'Judd-Vos XYZ'},'FontName','Helvetica','FontSize',20);
end
view([50 16]);
axis('square');
xlim([0 1.6]); ylim([0 1.6]); zlim([0 1.6]);
set(gca,'XTick',[0 0.8 1.6]);
set(gca,'YTick',[0 0.8 1.6]);
set(gca,'ZTick',[0 0.8 1.6]);
% saveas(gcf,fullfile(outputDir,'XYZSpace.tiff'),'tif');

% Plots of cones and XYZ
figure; clf; hold on
set(gca,'FontName','Helvetica','FontSize',16);
plot(wls,T_cones(1,:),'r','LineWidth',3);
plot(wls,T_cones(2,:),'g','LineWidth',3);
plot(wls,T_cones(3,:),'b','LineWidth',3);
legend({'L', 'M', 'S'},'FontName','Helvetica','FontSize',12);
xlabel('Wavelength (nm)','FontName','Helvetica','FontSize',20);
ylabel('Normalized Sensitivity','FontName','Helvetica','FontSize',20);
title(whichCones);
xlim([400 700]);
saveas(gcf,fullfile(outputDir,'LMSFundamentals.tiff'),'tif');

figure; clf; hold on
set(gca,'FontName','Helvetica','FontSize',16);
plot(wls,T_XYZ(1,:),'r','LineWidth',3);
plot(wls,T_XYZ(2,:),'g','LineWidth',3);
plot(wls,T_XYZ(3,:),'b','LineWidth',3);
legend({'X', 'Y', 'Z'},'FontName','Helvetica','FontSize',12);
xlabel('Wavelength (nm)','FontName','Helvetica','FontSize',20);
ylabel('CMF Value','FontName','Helvetica','FontSize',20);
if (strcmp(whichCones,'StockmanSharpe'))
    title('CIE Physiological XYZ','FontName','Helvetica','FontSize',20);
elseif (strcmp(whichCones,'SmithPokorny'))
    title({'Judd-Vos XYZ'},'FontName','Helvetica','FontSize',20);
end
xlim([400 700]);
% saveas(gcf,fullfile(outputDir,'XYZColorMatchingFcns.tiff'),'tif');

%% Get info to extract data from Danilova and Mollon 2025
% Generate unit circle
ell_pts = 1000;
t = linspace(0, 2*pi, ell_pts);
circle = [cos(t); sin(t)];  % 2xN matrix
%
%  Read in the ellipse table
ellipseDataTable = readtable('DanilovaMollon2025_Table1.xlsx','VariableNamingRule','preserve');
% stimulus size 
stim_size = [32, 48, 240]; %in min

% Figure out both coordinates from table.  The table gives the l coordinate
% of the reference, and tells us the angle of the line in the space it
% lives in.
refLCoords = ellipseDataTable.Ref_LminusM'; % Extract l coordinates from the table

% Loop through the table rows (one row per reference)
refSCoords = NaN(1, height(ellipseDataTable));
[ellipseAxis1, ellipseAxis2, ellipseAngle] = deal(NaN(height(ellipseDataTable), length(stim_size)));
[rotMat, stretchMat, ellMat] = deal(NaN(height(ellipseDataTable), length(stim_size), 2, 2));
ellContour = NaN(height(ellipseDataTable), length(stim_size), 2, ell_pts);
for rr = 1:height(ellipseDataTable)
    % If both l and s are there, just read them out, otherwise compute.
    if (ellipseDataTable.Ref_S(rr) == -1)
        % Need to figure out reference s coord
        deltaL = refLCoords(rr) - lsYD65(1);
        deltaS = deltaL*tand(ellipseDataTable.LineAngle(rr));
        refSCoords(rr) = lsYD65(2) + deltaS;
    else
        % We have reference s coord already
        refSCoords(rr) = ellipseDataTable.Ref_S(rr);
    end

    for ll = 1:length(stim_size)
        eval(sprintf('ellipseAxis1(rr,ll) = ellipseDataTable.L1_%d(rr);', stim_size(ll)));
        eval(sprintf('ellipseAxis2(rr,ll) = ellipseDataTable.L2_%d(rr);',  stim_size(ll)));
        eval(sprintf('ellipseAngle(rr,ll) = ellipseDataTable.Theta_%d(rr);',  stim_size(ll)));

        rotMat(rr,ll,:,:) = [cosd(ellipseAngle(rr,ll)), -sind(ellipseAngle(rr,ll)); ...
                             sind(ellipseAngle(rr,ll)), cosd(ellipseAngle(rr,ll))];
        stretchMat(rr,ll,:,:) = [ellipseAxis1(rr,ll)^2, 0;...
                              0, ellipseAxis2(rr,ll)^2];
        ellMat(rr,ll,:,:) = squeeze(rotMat(rr,ll,:,:)) * squeeze(stretchMat(rr,ll,:,:)) * squeeze(rotMat(rr,ll,:,:))';
    
        % Transform unit circle into ellipse using Cholesky or sqrtm
        ellContour(rr,ll,:,:) = sqrtm(squeeze(ellMat(rr,ll,:,:))) * circle;
    end
end

%% plot the ellipses
scaler_vis = [1,1,4];
y_bds = [0, 0.2];
x_bds = [0.58, 0.78];
% Plot
figure;
for ll = 1:length(stim_size)
    subplot(1,length(stim_size), ll)
    %horizontal line
    plot(x_bds, ones(1,2).*lsYD65(2), 'k:'); hold on
    %vertical line
    plot(lsYD65(1).*ones(1,2), y_bds, 'k:');
    %45 deg line
    plot(lsYD65(1) + [1, -1], lsYD65(2) + [tand(45),-tand(45)], 'k--');
    %135 deg line
    plot(lsYD65(1) + [1, -1], lsYD65(2) + [tand(135),-tand(135)], 'k--');
    %165 deg line
    plot(lsYD65(1) + [1, -1], lsYD65(2) + [tand(165), -tand(165)], 'k-');
    for i = 1:height(ellipseDataTable)
        plot(refLCoords(i) + scaler_vis(ll).*squeeze(ellContour(i,ll,1,:)),...
             refSCoords(i) + scaler_vis(ll).*squeeze(ellContour(i,ll,2,:)), 'k', 'LineWidth', 2);
    end
    axis square; hold off
    xlabel('L/(L+M)'); ylabel('S/(L+M)');
    xlim(x_bds); ylim(y_bds);
    xticks(0.6:0.05:0.75); yticks(0:0.05:0.2);
    title(sprintf('D = %d min', stim_size(ll)));
end
set(gcf,'Unit','Normalized','Position',[0, 0, 0.7,0.3]);

%% convert the ref to DKL
cal_path = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/';
matfileName = 'Transformation_btw_color_spaces.mat';
outputName = fullfile(cal_path,matfileName);
CalData = load(outputName);

%obtain transformation matrices
M_LMSTORGB = CalData.DELL_02242025_texture_right.M_LMSTORGB;
M_RGBTo2DW = CalData.DELL_02242025_texture_right.M_RGBTo2DW;

%get the monitor's primary
P_device = CalData.DELL_02242025_texture_right.cal.processedData.P_device;
%grab the previous S for splining
previous_S = CalData.DELL_02242025_texture_right.cal.rawData.S; 
%spline it to be consistent with what we have done so far
P_device_new = SplineSpd(previous_S, P_device, S);
%calculate luminance
T_lum = T_Y * (P_device_new * [0.5, 0.5, 0.5]');

[LMS_ref, RGB_ref, W_ref] = deal(NaN(height(ellipseDataTable), 3));

for rr = 1:height(ellipseDataTable)
    %here if we call MacBoynToLMS_v2, T_lum should be L'+M'
    %if we call MacBoynToLMS, T_lum should be L+M
    LMS_ref(rr,:) = MacBoynToLMS_v2(inv(mollonScaleMatrix)*[refLCoords(rr), refSCoords(rr)]', T_cones, T_Y, T_lum);

    %sanity check
    recover_ref = LMSToMacBoyn(LMS_ref(rr,:)', T_cones, T_Y, 1);
    if sum(recover_ref - [refLCoords(rr), refSCoords(rr)]) > 1e-4; error('We failed to recover the reference stimuli.'); end

    RGB_ref(rr,:) = M_LMSTORGB * LMS_ref(rr,:)';
    W_ref(rr,:) = M_RGBTo2DW * RGB_ref(rr,:)';
end

%display (if everything works fine, the 3rd column of W should be roughly 1)
W_ref






