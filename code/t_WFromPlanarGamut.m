% t_WFromPlanarGamut
%
% Try to figure out how we take (e.g.) the isoluminant plane
% and put a [-1,1] x [-1,1] coordinate system on the region
% of it that is within the gamut of a monitor.

%% Initialize
clear; close all;

%% Retrieve the correct calibration file
whichCalFile = 'NEC_08092024.mat';
whichCalNumber = 1;
nDeviceBits = 14;
whichCones = 'ss2';
cal = LoadCalFile(whichCalFile,whichCalNumber,getpref('BrainardLabToolbox','CalDataFolder'));

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

%% Compute ambient
ambientLMS = SettingsToSensor(calObjCones,[0 0 0]');
ambientXYZ = SettingsToSensor(calObjXYZ,[0 0 0']');

%% Compute the background, taking quantization into account
%
% The calculations here account for display quantization.
SPECIFIEDBG = true;
if (SPECIFIEDBG)
    bgxyYTarget = [0.31, 0.31, 40]';
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
%
% Get matrix that transforms between incremental
% cone coordinates and DKL coordinates 
% (Lum, RG, S).
[M_ConeIncToDKL,LMLumWeights] = ComputeDKL_M(bgLMS,T_cones,T_Y);
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
fprintf('Pooled cone contrast for unit DKL directions with initial scaling: %0.3g %0.3g %0.3g\n', ...
    lumPooled, rgPooled, sPooled);

%% Max contrast
% initialize
nAngles = 1000;
[gamutSettings, gamutLMS, gamutContrast, gamutDKL] = deal(NaN(3, nAngles));
vectorLengthGamutContrast = NaN(1, nAngles);

% Find maximum in gamut contrast for a set of color directions.
% This calculation does not worry about quantization.  It is
% done for the main case.  Because we care about this primarily
% for the XXX experiment, since the experimental specification
% matched what we intended for tha experiment.
theAngles = linspace(0,2*pi,nAngles);
gamut_bg_primary = NaN(3, nAngles);
for aa = 1:nAngles
    % Get a unit contrast vector at the specified angle
    targetDKLDir = [0 cos(theAngles(aa)) sin(theAngles(aa))]';

    % Convert from DKL to cone contrast to cone excitation direction.
    % Don't care about length here as that is handled by the contrast
    % maximization code below.
    theLMSExcitations = ContrastToExcitation(M_ConeIncToConeContrast*M_DKLToConeInc*targetDKLDir,bgLMS);

    % Convert the direction to the desired direction in primary space.
    % Since this is desired, we do not go into settings here. Adding
    % and subtracting the background handles the ambient correctly.
    thePrimaryDir = SensorToPrimary(calObjCones,theLMSExcitations) - SensorToPrimary(calObjCones,bgLMS);

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
figure
%add walls
fill3([1,0,0,1],[0,0,0,0],[0,0,1,1],'k','FaceAlpha',0.05); hold on
fill3([0,0,0,0],[1,0,0,1],[0,0,1,1],'k','FaceAlpha',0.05); 
fill3([1,0,0,1],[1,1,1,1],[0,0,1,1],'k','FaceAlpha',0.05); 
fill3([1,1,1,1],[1,0,0,1],[0,0,1,1],'k','FaceAlpha',0.05); 
%real plots
scatter3(bgPrimary(1),bgPrimary(2),bgPrimary(3),200,'g+','lineWidth',5);  
scatter3(gamut_bg_primary(1,:), gamut_bg_primary(2,:), gamut_bg_primary(3,:),...
    20, 'k', 'filled', 'MarkerEdgeColor','g'); 
fill3(gamut_bg_primary(1,:), gamut_bg_primary(2,:), gamut_bg_primary(3,:),...
    'k','FaceColor','k','FaceAlpha',0.3);
for aa = 1:25:nAngles
    vec_aa = horzcat(bgPrimary, gamut_bg_primary(:,aa));
    plot3(vec_aa(1,:), vec_aa(2,:), vec_aa(3,:),'k-.');
end
xlim([0,1]); ylim([0,1]); zlim([0,1]); xlabel('R'); ylabel('G'); zlabel('B')
axis square; grid on

%%  Make a plot of the gamut in the DKL isoluminant plane
figure; clf; hold on;
plot(gamutDKL(2,:),gamutDKL(3,:),'k','LineWidth',2);
axis('square');
xlabel('DKL L/(L+M)')
ylabel('DKL S');

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
        plot(intersectingPoints(2,ll),intersectingPoints(3,ll),...
            'bo','MarkerFaceColor','b','MarkerSize',14);
        plot(intersectingPoints1(2,ll),intersectingPoints1(3,ll),...
            'ro','MarkerFaceColor','r','MarkerSize',10);
    else
        corner(ll) = false;
    end
end

%% Select out the corner coordinates in 2D
cornerIndices = find(corner);
cornerPointsDKLPlane = intersectingPoints1(2:3,cornerIndices);
bgDKLPlane = [bgDKL(2:3)];
bgDKLPlane_ext = [bgDKLPlane; 0];
numCor = length(cornerIndices);
use_builtInTrans = true;

% If there are 4 corners, try to map to W space.
if (length(cornerIndices) == 4)
    targetCorners = [ [-1 -1]' [-1 1]' [1 -1]' [1 1]' ];
    targetCorners_ext = vertcat(targetCorners, ones(1,numCor));
    cornerPointsDKLPlane_ext = vertcat(cornerPointsDKLPlane, ones(1,numCor));
    if use_builtInTrans
        tform = fitgeotform2d(cornerPointsDKLPlane'-bgDKLPlane',...
            targetCorners',"projective");
        M_DKLPlaneTo2DW = tform.A;
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
    %M_2DWToDLKPlane = inv(M_DKLPlaneTo2DW);
    cornerPoints2DW_temp = M_DKLPlaneTo2DW*(cornerPointsDKLPlane_ext- bgDKLPlane_ext);
    cornerPoints2DW = cornerPoints2DW_temp(1:2,:) ./ cornerPoints2DW_temp(3,:);
    gamut2DW_temp = M_DKLPlaneTo2DW*([gamutDKLPlane;ones(1,nAngles)]-bgDKLPlane_ext);
    gamut2DW = gamut2DW_temp(1:2,:)./gamut2DW_temp(3,:);
    
    % Make a plot of the gamut in the DKL isoluminantplane
    figure; clf; hold on;
    plot(gamut2DW(1,:),gamut2DW(2,:),'k','LineWidth',2);
    for cc = 1:length(cornerIndices)
        plot(cornerPoints2DW(1,cc),cornerPoints2DW(2,cc),'bo','MarkerFaceColor','b','MarkerSize',14);
    end
    xlim([-1.5 1.5])
    ylim([-1.5 1.5]);
    axis('square');
    xlabel('W space dim 1');
    ylabel('W space dim 2');
end