
analysisDir = getpref('ColorEllipsoids', 'ELPSAnalysis');
myFigDir = 'InitialExploration';
outputDir = fullfile(analysisDir,myFigDir);
if (~exist('outputDir'))
    mkdir(outputDir);
end

outputName = fullfile(outputDir,'whateverThisIs.ext');
