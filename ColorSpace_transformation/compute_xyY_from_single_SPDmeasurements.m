%% Initialize
clear all; close all; clc

%Retrieve the correct calibration file
whichCalFile = 'DELL_10082025_background.mat';
whichCalNumber = 1;
nDeviceBits = 14; %doesn't have to be the true color depth; we can go higher
whichCones = 'ss2';
cal_path = fullfile(getpref('ColorEllipsoids','ELPSMaterials'),'Calibration');
%cal_path = '/Volumes/T9/Aguirre-Brainard Lab Dropbox/Fangfang Hong/ELPS_materials/Calibration/';
cal = LoadCalFile(whichCalFile,whichCalNumber,cal_path);
fig_output_path = fullfile(cal_path, 'Plots');
% Ensure the output folder exists
if ~exist(fig_output_path, 'dir')
    mkdir(fig_output_path);
end

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
wvl = SToWls(Scolor);

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
end
SetSensorColorSpace(calObjCones,T_cones,Scolor);

%% XYZ cal object
calObjXYZ = ObjectToHandleCalOrCalStruct(cal);

% Get gamma correct
CalibrateFitGamma(calObjXYZ, nDeviceLevels);
SetGammaMethod(calObjXYZ,gammaMode);

% XYZ
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

% Compute ambient
ambientXYZ = SettingsToSensor(calObjXYZ,[0 0 0']');

%% Compute the background, taking quantization into account
% The calculations here account for display quantization.
bgPrimary = SettingsToPrimary(calObjCones,PrimaryToSettings(calObjCones,[0.615 0.615 0.615]')); 
%0.27, 0.29, 0.32, 0.30, 0.31
%0.63, 0.61, 0.62, 0.615
bgXYZ = PrimaryToSensor(calObjXYZ,bgPrimary);
bgxyY = XYZToxyY(bgXYZ);

%% Compute background xyY for various choices of  daylight CCT
fprintf('\nComputing possible adapting backgrounds\n')
load B_cieday.mat
B_cieday = SplineSpd(S_cieday,B_cieday,Scolor);
adaptingCCTs = [4000 6500 12000, 20000];
for cct = 1:size(adaptingCCTs,2)
    % Generate CIE daylight with specified correlated color temperature (CCT)
    spdCCTs(:,cct) = GenerateCIEDay(adaptingCCTs(cct),B_cieday);

    % Get XYZ and xyY
    XYZCCTs(:,cct) = T_xyz*spdCCTs(:,cct);
    xyYCCTs(:,cct) = XYZToxyY(XYZCCTs(:,cct));

    % Swap in Y from the background.  Need to do this in chromaticity
    xyYCCTs(3,cct)= bgxyY(3);
    XYZCCTs(:,cct) = xyYToXYZ(xyYCCTs(:,cct));

    % Get linear RGB.  I think, let's not quantize here and just get the continuous value
    % in linear RGB.
    cctPrimary(:,cct) = SensorToPrimary(calObjXYZ,XYZCCTs(:,cct));
    LMSCCTs(:,cct) = PrimaryToSensor(calObjCones,cctPrimary(:,cct));

    % Print out
    fprintf('\tCCT: %d;\tlinear rgb; %0.4f; %0.4f; %0.4f;\txyY: %0.4f, %0.4f, %0.2f;\tLMS: %0.4f, %0.4f, %0.4f\n', ...
        adaptingCCTs(cct), cctPrimary(1,cct), cctPrimary(2,cct), cctPrimary(3,cct), ...
        xyYCCTs(1,cct), xyYCCTs(2,cct), xyYCCTs(3,cct), LMSCCTs(1,cct), LMSCCTs(2,cct), LMSCCTs(3,cct));
end

% Print basic info and report on monitor
fprintf('\nCalibration file %s, calibration number %d, calibration date %s\n', ...
    whichCalFile,whichCalNumber,calObjXYZ.cal.describe.date);
fprintf('\nBackground x,y = %0.4f, %0.4f\n',bgxyY(1),bgxyY(2));
fprintf('Background Y = %0.2f cd/m2, ambient %0.3f cd/m2\n',bgXYZ(2),ambientXYZ(2));

%% load measurement
pattern = fullfile(cal_path, 'single_measurements_SPD', 'Measurement_background_12000K*.mat');

% List matches
files = dir(pattern);

if isempty(files)
    error('No files match: %s', pattern);
end

% Option A: pick the most recent file
[~, idx] = max([files.datenum]);   % or use [files.datenum] in older MATLAB
match_path = fullfile(files(idx).folder, files(idx).name);

load(match_path);

%get the spectral power distribution
spd_background = SplineCmf(theSpectralSupport', theSPD, wvl)*2;

%from SPD to XYZ
XYZ_background = T_xyz * spd_background';
xyY_background = XYZToxyY(XYZ_background);

%print out
fprintf('\nThe xyY of the cubic room: %0.4f, %0.4f, %0.4f.\n', xyY_background(1), xyY_background(2), xyY_background(3));

