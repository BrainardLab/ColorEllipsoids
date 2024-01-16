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
sysInfo = GetComputerInfo();
switch (sysInfo.localHostName)
    otherwise
        % Some unspecified machine, try user specific customization
        switch(sysInfo.userShortName)
            % Could put user specific things in, but at the moment generic
            % is good enough
                
            otherwise
                if ismac
                    dbJsonConfigFile = '~/.dropbox/info.json';
                    fid = fopen(dbJsonConfigFile);
                    raw = fread(fid,inf);
                    str = char(raw');
                    fclose(fid);
                    val = jsondecode(str);
                    baseDir = val.business.path;
                end
        end
end

%% Project prefs
setpref(projectName,'LEDSpectraDir',fullfile(baseDir,'COLE_materials','JandJProjector','LEDSpectrumMeasurements'));

% Calibration
setpref('BrainardLabToolbox','CalDataFolder',fullfile(baseDir,'COLE_materials','Calibration'));

% Data dir
setpref(projectName,'TestDataFolder',fullfile(baseDir,'COLE_datadev','TestData'));

% Main experiment data dir (as of 10/14/22)
setpref(projectName,'COLEData',fullfile(baseDir,'COLE_data'));

% Experiment analysis data dir (as of 10/19/22)
setpref(projectName,'COLEAnalysis',fullfile(baseDir,'COLE_analysis'));

% Check data dir (This is for screen stability and channel additivity data)
setpref(projectName,'CheckDataFolder',fullfile(baseDir,'COLE_materials','JandJProjector','CheckData'));
setpref(projectName,'CheckDataFolderSACC',fullfile(baseDir,'COLE_materials','FromSACCMeasurements','JandJProjector','CheckData'));

% COLE materials.
setpref(projectName,'COLEMaterials',fullfile(baseDir,'COLE_materials'));

% David's melanopsion work
setpref(projectName,'COLEMelanopsin',fullfile(baseDir,'COLE_melanopsin'));

% New experiment analysis data dir (as of 10/03/23).
setpref(projectName,'COLEAnalysisRefit',fullfile(baseDir,'COLE_analysis_refit'));

% We will save the final results in this directory (as of 10/13/23).
setpref(projectName,'COLEAnalysisFinal',fullfile(baseDir,'COLE_analysis_final'));

