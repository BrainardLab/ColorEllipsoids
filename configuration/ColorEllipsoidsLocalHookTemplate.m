function ColorEllipsoidsLocalHook
% ColorEllipsoidsLocalHook
%
% Configure things for working on the ColorEllipsoids project.
%
% For use with the ToolboxToolbox.  If you copy this into your
% ToolboxToolbox localToolboxHooks directory (by default,
% ~/localToolboxHooks) and delete "LocalHooksTemplate" from the filename,
% this will get run when you execute tbUseProject('ColorEllipsoids') to set up for
% this project.  You then edit your local copy to match your configuration.
%
% You will need to edit the project location and i/o directory locations
% to match what is true on your computer.

%% Define project
projectName = 'ColorEllipsoids';

%% Say hello
fprintf('Running %s local hook\n',projectName);

%% Clear out old preferences
if (ispref(projectName))
    rmpref(projectName);
end

%% Specify project location
projectBaseDir = tbLocateProject(projectName);

% If we ever needed some user/machine specific preferences, this is one way
% we could do that.
if ismac
    dbJsonConfigFile = '~/.dropbox/info.json';
    fid = fopen(dbJsonConfigFile);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);
    baseDir = val.business.path;
else
    error('Need to configure how to find the data base directory for the computer you are on');
end
 
%% Project prefs
setpref(projectName,'LEDSpectraDir',fullfile(baseDir,'ELPS_materials','JandJProjector','LEDSpectrumMeasurements'));

% Calibration
setpref('BrainardLabToolbox','CalDataFolder',fullfile(baseDir,'ELPS_materials','Calibration'));

% Data dir
setpref(projectName,'TestDataFolder',fullfile(baseDir,'ELPS_datadev','TestData'));

% Main experiment data dir (as of 10/14/22)
setpref(projectName,'ELPSData',fullfile(baseDir,'ELPS_data'));

% Experiment analysis data dir (as of 10/19/22)
setpref(projectName,'ELPSAnalysis',fullfile(baseDir,'ELPS_analysis'));

% ELPS materials.
setpref(projectName,'ELPSMaterials',fullfile(baseDir,'ELPS_materials'));



