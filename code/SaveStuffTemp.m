
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'Simulation_DataFiles';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end
outputName = fullfile(outputDir,'whateverThisIs.ext');
