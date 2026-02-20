%% Initialize
%{
This script supports the color-adaptation experiment by generating candidate
CIE daylight adapting backgrounds (parameterized by correlated color
temperature, CCT), converting them into monitor settings using a chosen
calibration, and comparing predicted chromaticities against CR-250
measurements for both the uniform background and the cubic room.

Main steps
----------
1) Select calibration by the context
   - Choose whether we are targeting the uniform adapting background or the
     cubic-room surfaces.
   - Load the corresponding monitor calibration file (date + condition tag).

2) Build calibration objects in two sensor spaces
   - Cone-fundamental space (Stockman–Sharpe 2°) for reporting LMS cone
     excitations produced by candidate settings.
   - XYZ space (CIEPhys2 by default; optional CIE 1931) for chromaticity and
     luminance computations.

3) Sweep candidate daylight CCTs and compute predicted settings
   - Generate CIE daylight SPDs for a dense set of CCTs (e.g., 4500–13000 K).
   - Convert SPD -> XYZ -> xyY and force all candidates to match the target
     background luminance (replace Y with bgxyY(3)).
   - Convert desired XYZ -> linear device primaries (continuous linear RGB),
     and compute the corresponding LMS cone excitations.
   - Print per-CCT summaries for sanity checking.
   - Optionally save the sweep outputs (.mat) for downstream Python selection
     (e.g., geodesic-distance based selection of adaptation conditions).

4) Load CR-250 measurements for selected CCTs and compare to predictions
   - For a subset of measured CCTs, load the most recent measurement file
     (per context) and compute measured xyY.
   - Print predicted vs measured xyY (and the predicted RGB used).

5) Optional export and visualization
   - Optionally export predicted/measured xyY (plus RGB) to CSV for Python.
%}
clear all; close all; clc

% -------------------------------------------------------------------------
% Select the appropriate monitor calibration file for this stimulus context
% -------------------------------------------------------------------------
% Calibration file dates used in this project:
%   '10082025' : Round 1 adaptation experiment (calibration for background)
%   '02012026' : Round 2 adaptation experiment (calibration for the cubic room)
%   '02102026' : Round 2 adaptation experiment (calibration for the background)
%
% We choose which calibration to load based on the "context" we are
% generating CCTs for:
%   - 'background' : the adapting background itself (brighter)
%   - 'cubicroom'  : the textured cubic-room surfaces (dimmer than background)
%
% Choose the stimulus context to calibrate / generate CCTs for
context = 'cubicroom';   % Options: 'cubicroom' or 'background'

% Map context -> (calibration date, calibration condition string)
switch context
    case 'background'
        cal_date = '02102026';      % Calibration date for background condition
        cal_str  = 'background';    % Calibration label used in filename
    case 'cubicroom'
        cal_date = '02012026';      % Calibration date for cubic-room texture
        cal_str  = 'texture_right'; % Which textured surface / channel was calibrated
    otherwise
        error('Unrecognized context: %s', context);
end

% Construct the calibration filename (matches naming convention on disk)
whichCalFile   = sprintf('DELL_%s_%s.mat', cal_date, cal_str);

% Some Cal files contain multiple calibration entries; choose which one to load
whichCalNumber = 1;

% Assumed device quantization (in bits) used for converting primaries <-> settings
nDeviceBits    = 14;

% Cone fundamentals / observer set used downstream (e.g., Stockman-Sharpe 2-deg)
whichCones     = 'ss2';

% Load the calibration from the standardized calibration directory
cal_path = fullfile(getpref('ColorEllipsoids','ELPSMaterials'), 'Calibration');
cal      = LoadCalFile(whichCalFile, whichCalNumber, cal_path);

% Directory for saving any diagnostic plots produced by this script
fig_output_path = fullfile(cal_path, 'Plots');

% Create the output folder if it does not already exist
if ~exist(fig_output_path, 'dir')
    mkdir(fig_output_path);
end

%% cal object
% Calibration object in cone-fundamental space (LMS / cone excitations)
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

% Get wavelength sampling from the calibration (S = [start step nSamples]).
% Convert S-format to an explicit wavelength vector in nm.
Scolor = calObjCones.get('S');
wvl = SToWls(Scolor);

% Set the "sensor" space to cone fundamentals (Stockman-Sharpe 2-degree).
% This lets us compute cone excitations from device settings via:
%   SettingsToSensor(calObjCones, settings)
% after the sensor space is set.
load T_cones_ss2
T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,Scolor);
SetSensorColorSpace(calObjCones,T_cones,Scolor);

% Calibration object in XYZ space (for luminance and chromaticity)
calObjXYZ = ObjectToHandleCalOrCalStruct(cal);

% Reuse the same gamma correction settings for XYZ computations.
% (Separate object so we can swap sensor/color-matching functions cleanly.)
CalibrateFitGamma(calObjXYZ, nDeviceLevels);
SetGammaMethod(calObjXYZ,gammaMode);

% Load XYZ color matching functions / spectral sensitivities.
% Optionally use:
%   - CIE 1931 2-deg CMFs (classic standard)
%   - CIE "physiological" set (often smoother / more physically-based)
USE1931XYZ = false;
if (USE1931XYZ)
    load T_xyz1931.mat
    T_xyz = 683*SplineCmf(S_xyz1931,T_xyz1931,Scolor);
else
    load T_xyzCIEPhys2.mat
    T_xyz = 683*SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,Scolor);
end

% Convenience handle for the Y (luminance) row of the XYZ sensitivity matrix
T_Y = T_xyz(2,:);

% Set the XYZ sensor space so SettingsToSensor returns [X; Y; Z]
SetSensorColorSpace(calObjXYZ,T_xyz,Scolor);

% Compute the ambient light contribution in XYZ (display "black" level).
% This measures what the calibration predicts when device settings are zero.
ambientXYZ = SettingsToSensor(calObjXYZ,[0 0 0']');

%% Compute the background, taking quantization into account
% The calculations here account for display quantization.
switch context
    case 'background'
        linear_rgb_bg = [0.6014, 0.6200, 0.6234]';
    case 'cubicroom'
        linear_rgb_bg = [0.3, 0.3, 0.3]';
    otherwise
        error('Unrecognized context: %s', context);
end
bgPrimary = SettingsToPrimary(calObjCones,PrimaryToSettings(calObjCones, linear_rgb_bg));   
bgXYZ = PrimaryToSensor(calObjXYZ,bgPrimary);
bgxyY = XYZToxyY(bgXYZ);

%% Compute background xyY for various choices of  daylight CCT
fprintf('\nComputing possible adapting backgrounds\n')

% Load the CIE daylight basis functions and spline them to match the
% calibration wavelength sampling (Scolor). This ensures all spectral
% computations live on the same wavelength grid as the calibration.
load B_cieday.mat
B_cieday = SplineSpd(S_cieday,B_cieday,Scolor);

% Define the candidate correlated color temperatures (CCTs) to evaluate.
% Use finer spacing in the mid range and coarser spacing at high CCT.
adaptingCCTs = [4500:50:7550, 7600:100:13000];
numK = length(adaptingCCTs);

% Preallocate outputs:
%   spdCCTs   : daylight SPDs sampled on Scolor (nWavelengths x numK)
%   XYZCCTs   : tristimulus values (3 x numK)
%   xyYCCTs   : chromaticity + luminance (3 x numK)
%   cctPrimary: linear device primaries corresponding to desired XYZ (3 x numK)
%   LMSCCTs   : cone excitations produced by those primaries (3 x numK)
spdCCTs = NaN(size(T_cones, 2), numK);
[XYZCCTs, xyYCCTs, cctPrimary, LMSCCTs] = deal(NaN(3, numK));
for cct = 1:numK
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
    fprintf('\tCCT: %d;\tlinear rgb; %0.4f; %0.4f; %0.4f;\txyY: %0.4f, %0.4f, %0.4f;\tLMS: %0.4f, %0.4f, %0.4f\n', ...
        adaptingCCTs(cct), cctPrimary(1,cct), cctPrimary(2,cct), cctPrimary(3,cct), ...
        xyYCCTs(1,cct), xyYCCTs(2,cct), xyYCCTs(3,cct), LMSCCTs(1,cct), LMSCCTs(2,cct), LMSCCTs(3,cct));
end

% Print basic info and report on monitor
fprintf('\nCalibration file %s, calibration number %d, calibration date %s\n', ...
    whichCalFile,whichCalNumber,calObjXYZ.cal.describe.date);
fprintf('\nBackground x,y = %0.4f, %0.4f\n',bgxyY(1),bgxyY(2));
fprintf('Background Y = %0.2f cd/m2, ambient %0.3f cd/m2\n',bgXYZ(2),ambientXYZ(2));

% -------------------------------------------------------------------------
% Optional: save the CCT sweep outputs to a .mat file for downstream Python.
%
% Typical Python usage:
%   - Compute geodesic distances between RGB settings (e.g., CCT vs D65).
%   - Select a subset of CCTs whose distances increase approximately linearly
%     in both the "orange" and "blue" directions.
%   - Find an "orange-side" CCT that matches the D65->12000K distance
%     (for symmetry against a condition tested in round 1).
% -------------------------------------------------------------------------
flag_output_to_mat = false;
if flag_output_to_mat
    outFile = fullfile(cal_path, sprintf(['single_measurements_SPD/adapting_',...
        'backgrounds_CCT_from%d_to%d.mat'], adaptingCCTs(1), adaptingCCTs(end)));
    
    % Use -v7 for best compatibility with scipy.io.loadmat
    save(outFile, ...
        'adaptingCCTs', 'spdCCTs', 'XYZCCTs', 'xyYCCTs', 'cctPrimary', 'LMSCCTs', ...
        '-v7');
end

%% load measurements
%{
---------------------------------------------------------------------------
How these measurements were obtained (CR-250 workflow)
---------------------------------------------------------------------------
We take the *predicted* linear RGB settings (printed above for each CCT) and
then measure the resulting chromaticity with the CR-250.

In practice, the calibration-based prediction is not perfect, so the RGB
that *nominally* corresponds to a target CCT may not land exactly on the
desired (x,y) when measured. The procedure was:

1) Background matching (per target CCT)
   - For each target CCT (e.g., 4800 K), MATLAB predicts a target chromaticity
     (x_target, y_target) and a corresponding linear RGB (r_pred, g_pred, b_pred).
   - I then searched for an adjusted linear RGB (r_adj, g_adj, b_adj) such that
     the *measured* chromaticity of the uniform background best matched
     (x_target, y_target).
   - Example: the RGB that produced the best match to the 4800 K target might
     actually be the RGB that the code labels as 4650 K. But the file label
     is 4800 K!

2) Cubic-room matching (per condition)
   - For each condition, after the background chromaticity was matched,
     I searched for the cubic-room RGB settings that produced a measured
     chromaticity matching the *measured* chromaticity of the uniform background
     for that same condition.

The code below loads (a) the predicted values for each CCT from the sweep,
and (b) the most recent CR-250 measurement file for that CCT/context, and
prints + optionally exports predicted vs measured xyY.
%}

%the list of CCTs that I made measurements
slc_CCT = [4650, 4800, 5100, 5500, 5900, 6300, 6500, 6800, 7550, 8600, 10400, 12000, 13000];
nCCT = numel(slc_CCT);

% ----------------------------
% Preallocate
% ----------------------------
slc_rgb_all          = nan(nCCT, 3);   % linear RGB (R,G,B)
slc_pred_xyY_all     = nan(nCCT, 3);   % predicted xyY (x,y,Y)
slc_meas_xyY_all     = nan(nCCT, 3);   % measured xyY (x,y,Y)

for i = 1:nCCT
    % Find index of this CCT in your arrays
    idx_i = find(adaptingCCTs == slc_CCT(i), 1);
    if isempty(idx_i)
        error('slc_CCT(%d) = %dK not found in adaptingCCTs.', i, slc_CCT(i));
    end

    % linear rgb (assume cctPrimary is 3 x nCCTs)
    slc_rgb_all(i, :) = cctPrimary(:, idx_i)';

    % predicted xyY (assume xyYCCTs is 3 x nCCTs)
    slc_pred_xyY_all(i, :) = xyYCCTs(:, idx_i)';

    % Load most recent measurement file for this CCT
    pattern = fullfile(cal_path, 'single_measurements_SPD', ...
        sprintf('Measurement_%s_%05dK.mat', context, slc_CCT(i))); %background, cubicroom

    files = dir(pattern);
    if isempty(files)
        error('No files match: %s', pattern);
    end
    [~, idx_latest] = max([files.datenum]);
    match_path = fullfile(files(idx_latest).folder, files(idx_latest).name);

    S = load(match_path);

    % SPD -> XYZ -> xyY
    spd_background = SplineCmf(S.theSpectralSupport', S.theSPD, wvl) * 2;
    XYZ_background = T_xyz * spd_background';
    xyY_background = XYZToxyY(XYZ_background);

    slc_meas_xyY_all(i, :) = xyY_background(:)';

    % ----------------------------
    % Print: CCT, linear RGB, predicted xyY, measured xyY
    % ----------------------------
    fprintf(['\nCCT %5dK | RGB = [%0.4f %0.4f %0.4f] | ', ...
             'pred xyY = [%0.4f %0.4f %0.4f] | meas xyY = [%0.4f %0.4f %0.4f]\n'], ...
        slc_CCT(i), ...
        slc_rgb_all(i,1), slc_rgb_all(i,2), slc_rgb_all(i,3), ...
        slc_pred_xyY_all(i,1), slc_pred_xyY_all(i,2), slc_pred_xyY_all(i,3), ...
        slc_meas_xyY_all(i,1), slc_meas_xyY_all(i,2), slc_meas_xyY_all(i,3));
end

% -------------------------------------------------------------------------
% Optional CSV export for Python plotting/selection
% Output columns:
%   - CCT_K
%   - linear_rgb      (stringified [R G B])
%   - predicted_xyY   (stringified [x y Y])
%   - measured_xyY    (stringified [x y Y])
%
% The CSV is intended for downstream Python analysis/plotting (e.g., compare
% predicted vs measured chromaticities, compute geodesic distances, choose
% a subset of adaptation conditions).
% -------------------------------------------------------------------------
rgb_str      = arrayfun(@(k) sprintf('[%0.6f %0.6f %0.6f]', slc_rgb_all(k,:)),      (1:nCCT)', 'UniformOutput', false);
pred_xyY_str = arrayfun(@(k) sprintf('[%0.6f %0.6f %0.6f]', slc_pred_xyY_all(k,:)), (1:nCCT)', 'UniformOutput', false);
meas_xyY_str = arrayfun(@(k) sprintf('[%0.6f %0.6f %0.6f]', slc_meas_xyY_all(k,:)), (1:nCCT)', 'UniformOutput', false);

flag_output_csv = false;
if flag_output_csv
    T = table(slc_CCT(:), rgb_str, pred_xyY_str, meas_xyY_str, ...
        'VariableNames', {'CCT_K','linear_rgb','predicted_xyY','measured_xyY'});
    out_csv = fullfile(cal_path, sprintf('single_measurements_SPD/%s_CCT_rgb_pred_meas_xyY.csv', context));
    writetable(T, out_csv);
    fprintf('\nWrote CSV: %s\n', out_csv);
end

%% optional (compare the chromaticity of the cubic room and the uniform  background)
% -------------------------------------------------------------------------
% Load measured xyY (background vs cubic room) from CSVs, if they exist.
% If either file is missing, skip plotting gracefully.
% -------------------------------------------------------------------------

out_csv1 = fullfile(cal_path, 'single_measurements_SPD/background_CCT_rgb_pred_meas_xyY.csv');
out_csv2 = fullfile(cal_path, 'single_measurements_SPD/cubicroom_CCT_rgb_pred_meas_xyY.csv');

has1 = exist(out_csv1, 'file') == 2;
has2 = exist(out_csv2, 'file') == 2;

if ~(has1 && has2)
    % Print a clear message and do nothing
    if ~has1
        fprintf('Skipping plot: missing file: %s\n', out_csv1);
    end
    if ~has2
        fprintf('Skipping plot: missing file: %s\n', out_csv2);
    end
else
    % ----------------------------
    % Read CSVs
    % ----------------------------
    T1 = readtable(out_csv1, 'Delimiter', ',', 'TextType', 'string');
    T2 = readtable(out_csv2, 'Delimiter', ',', 'TextType', 'string');

    % ----------------------------
    % Parse measured_xyY strings: "[x y Y]" -> numeric array (n x 3)
    % ----------------------------
    s1 = cellstr(T1.measured_xyY);
    meas_xyY1 = cell2mat(cellfun(@(s) sscanf(s, '[%f %f %f]').', s1, 'UniformOutput', false));

    s2 = cellstr(T2.measured_xyY);
    meas_xyY2 = cell2mat(cellfun(@(s) sscanf(s, '[%f %f %f]').', s2, 'UniformOutput', false));

    % ----------------------------
    % Basic sanity check: same number of rows (CCTs)
    % If not, plot only matched rows to avoid indexing errors.
    % ----------------------------
    n1 = size(meas_xyY1, 1);
    n2 = size(meas_xyY2, 1);
    n  = min(n1, n2);
    if n1 ~= n2
        fprintf('Warning: row mismatch (background=%d, cubicroom=%d). Plotting first %d rows.\n', n1, n2, n);
        meas_xyY1 = meas_xyY1(1:n, :);
        meas_xyY2 = meas_xyY2(1:n, :);
    end

    % ----------------------------
    % Plot xy chromaticities and connect matched conditions
    % ----------------------------
    figure; hold on;
    f1 = scatter(meas_xyY1(:,1), meas_xyY1(:,2), 80, 'r');
    f2 = scatter(meas_xyY2(:,1), meas_xyY2(:,2), 80, 'b');

    for i = 1:n
        plot([meas_xyY1(i,1), meas_xyY2(i,1)], ...
             [meas_xyY1(i,2), meas_xyY2(i,2)], ...
             'Color', 'k', 'LineWidth', 1.5);
    end

    xlim([0.25, 0.37]);
    ylim([0.26, 0.38]);
    axis equal; grid on;
    xlabel('CIE x');
    ylabel('CIE y');
    legend([f1, f2], {'background', 'cubic room'}, 'Location', 'best');
end