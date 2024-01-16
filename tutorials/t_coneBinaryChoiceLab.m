%% t_coneBinaryChoiceLab

%% Initialize
clear; close all;
verbose = true;

%% Define wavelength sampling
%
% We'll spline all spectral functions to this common
% wavlength axis after loading them in.
S = [400 5 61];
wls = SToWls(S);

%% Load in LMS cone fundamentals
%
% Data in Psychtoolbox
load T_cones_ss2.mat
T_cones = SplineCmf(S_cones_ss2,T_cones_ss2,S);

%% Load in XYZ CMFs 
%
% Data in Psychtoolbox
load T_xyzCIEPhys2.mat
T_xyz = SplineCmf(S_xyzCIEPhys2,T_xyzCIEPhys2,S);

%% Load in primaries for a monitor.
%
% We'll use this to define a reasonable color gamut,
% as well as a white point for CIELAB conversions.
%
% These primaries are from a Psychtoolbox example file.
load B_monitor.mat
B_monitor = SplineSpd(S_monitor,B_monitor,S);

% Sometimes we want to go from LMS to monitor coordinates
M_RGBToLMS = T_cones*B_monitor;
M_LMSTORGB = inv(M_RGBToLMS);

%% Get matrix that converts between LMS and XYZ
M_LMSToXYZ = ((T_cones)'\(T_xyz)')';
if (verbose)
    T_xyzCheck = M_LMSToXYZ*T_cones;
    figure; clf; hold on;
    plot(wls,T_xyz,'k','LineWidth',4);
    plot(wls,T_xyzCheck,'r','LineWidth',2);
    xlabel('Wavelength (nm)');
    ylabel('XYZ Tristimulus Value');
end

%% Set up Weibull psychometric function on CIELAB deltaE 
% 
% CIELAB defines a difference deltaE between any two
% points in color space.  A deltaE of 1 is more or less
% forced choice detection threshold, and a deltaE of 3
% is close to perfectly discriminated.
% We'll use these two facts to pin down the parameters of
% a Weibull function, and then later use those parameters
% to predict binary choice probabilities given deltaE.
%
% The PTB fit functions are not as robust as those in packages
% such as Palamedes, but here we just need two parameters once,
% and can get a good fit by futzing with the initial guesses
% by hand.
nTrialSimulateForFit = 100;
deltaEsForFit = [0 1 2 3];
probCsForFit = [0.5 0.65 0.92 0.99];
alpha0 = 2;
beta0 = 3.5;
[alpha1] = FitWeibAlphTAFC(deltaEsForFit,round(nTrialSimulateForFit*probCsForFit),round(nTrialSimulateForFit*(1-probCsForFit)),alpha0,beta0);
[alpha,beta] = FitWeibTAFC(deltaEsForFit,round(nTrialSimulateForFit*probCsForFit),round(nTrialSimulateForFit*(1-probCsForFit)),alpha1,beta0);

% Check that the fit is OK
if (verbose)
    deltaEsForPlot = linspace(0,5,1000);
    probCsForPlot = ComputeWeibTAFC(deltaEsForPlot,alpha,beta);
    figure; clf; hold on;
    plot(deltaEsForFit,probCsForFit,'ro','MarkerSize',10,'MarkerFaceColor','r');
    plot(deltaEsForPlot,probCsForPlot,'r','LineWidth',2);
    xlabel('Delta E');
    ylabel('Prob Correct');
end

%% Define some stimuli 
%
% Start by getting monitor gray point.  B_monitor
% describes an old CRT that was rather blue in its
% color balance, so we tweak from the natural [0.5 0.5 0.5]
% to make it come out gray in the end
backgroundRGB = [0.5 0.4 0.37]';
backgroundSpd = B_monitor*backgroundRGB;
backgroundLMS = T_cones*backgroundSpd;
backgroundXYZ = M_LMSToXYZ*backgroundLMS;

%% Pick a reference point in color space.
%
% We will define a threshold contour around this point.
% Choosing the reference in the monitor space guarantees
% we will be in gamut [0-1].  We then convert that to
% LMS coordinates, because we want to think in LMS
% space for the most part.
referenceRGB = [0.6 0.5 0.4]';
referenceSpd = B_monitor*referenceRGB;
referenceLMS = T_cones*referenceSpd;

%% Pick a color direction in LMS space
%
% We will compute a psychometric function for
% perturbations around the reference in this color
% direction.
%
% It's hard to intuit the scale of perturbations
% in LMS.  But it is easy to figure out how
% far we can go before we hit the edge of the monitor
% gamut.  We use that to define the scale we'll
% explore.
rawDeltaLMS = [1 0 0]';
rawDeltaRGB = M_LMSTORGB*rawDeltaLMS;
gamutScalar = MaximizeGamutContrast(rawDeltaRGB,referenceRGB);
deltaLMS = gamutScalar*rawDeltaLMS;
deltaRGB = gamutScalar*rawDeltaRGB;
if (verbose)
    % All of the values here should be between 0 and 1,
    % and at least one should be 0 or 1
    max(referenceRGB + deltaRGB)
    min(referenceRGB + deltaRGB)
    max(referenceRGB - deltaRGB)
    max(referenceRGB - deltaRGB)
end

%% We now have a scaled deltaLMS
%
% This has the property that when we add it to the reference
% after scaling by a number between -1 and 1, we are within
% the monitor gamut.  The positive and negative perturbations
% are in the opposite directions around the white point.  We
% will illustrate computing the psychometric function for
% positive perturbations.
nDeltaScalars = 1000;
deltaScalars = linspace(0,1,nDeltaScalars);

%% Loop over the perturbations and compute prob correct for each one
%
% But first convert the reference stimulus to Lab, just once.
% This computation depends on an adaptation 'white' point,
% which here we take as the monitor backround XYZ.
referenceXYZ = M_LMSToXYZ*referenceLMS;
referenceLab = XYZToLab(referenceXYZ,backgroundXYZ);

for dd = 1:nDeltaScalars
    % Get comparison Lab for this scalar
    comparisonLMS = referenceLMS + deltaScalars(dd)*deltaLMS;
    comparisonXYZ = M_LMSToXYZ*comparisonLMS;
    comparisonLab = XYZToLab(comparisonXYZ,backgroundXYZ);

    % DeltaE is just the L2 norm of the Lab difference
    deltaEs(dd) = norm(comparisonLab-referenceLab,2);

    % Prob correct is obtained from the Weibull we fit above
    probCs(dd) = ComputeWeibTAFC(deltaEs(dd),alpha,beta);
end

% Plot the psychometric function points we computed here
% for the specified delta scalars
if (verbose)
    % It's not interesting to plot a lot of 1s, so we set
    % the x-axis plot limit at a reasonable deltaE value
    maxDeltaEToPlot = 6;

    figure; clf; hold on;
    plotPointSpacing = 10;
    plot(deltaEs(1:plotPointSpacing:end),probCs(1:plotPointSpacing:end),'ro','MarkerSize',8,'MarkerFaceColor','r');
    plot(deltaEsForPlot,probCsForPlot,'r','LineWidth',2);
    xlim([0 maxDeltaEToPlot]);
    xlabel('Delta Scalar');
    ylabel('Prob Correct');
    title({sprintf('Reference LMS: %0.2f %0.2f %0.2f; Delta LMS: %0.2f %0.2f %0.2f', ...
        referenceLMS(1),referenceLMS(2),referenceLMS(3),deltaLMS(1),deltaLMS(2),deltaLMS(3)); ' '});
end

%% We might reasonably want to get a feel for what these stimuli.  
%
% Today's displays do something close to the sRGB standard, and
% we can go from XYZ to that standard pretty easily.  The only
% trick is to get our XYZ units scaled into those of the sRGB
% standard.
threshLevels = [0.6 0.7 0.8 0.9];
for tt = 1:length(threshLevels)
    % Find compute probCorrect as close as possible to the specified level.
    [~,minIndex] = min(abs(probCs-threshLevels(tt)));
    threshDeltaScalars(tt) = deltaScalars(minIndex(1));
    threshProbCs(tt) = probCs(minIndex(1));
    
    % ComparisonLMS
    threshComparisonLMS(:,tt) = referenceLMS + threshDeltaScalars(tt) * deltaLMS;
end
if (max(abs(threshProbCs-threshLevels)) > 0.005)
    error('Have not spaced deltaScalars finely enough to obtain desired threshold levels precisely');
end
threshComparisonXYZ = M_LMSToXYZ*threshComparisonLMS;

% Convert to linear sRGB
backgroundSRGBPrimary = XYZToSRGBPrimary(backgroundXYZ);
referenceSRGBPrimary = XYZToSRGBPrimary(referenceXYZ);
threshComparisonSRGBPrimary = XYZToSRGBPrimary(threshComparisonXYZ);
allSRGBPrimary = [backgroundSRGBPrimary(:) ; referenceSRGBPrimary(:) ; threshComparisonSRGBPrimary(:)];
if (any(allSRGBPrimary < 0))
    error('Flying too close to the one.  At least one sRGB primary value is negative');
end
sRGBPrimaryFactor = 1/max(allSRGBPrimary);
backgroundSRGB = SRGBGammaCorrect(backgroundSRGBPrimary*sRGBPrimaryFactor,false);
referenceSRGB = SRGBGammaCorrect(referenceSRGBPrimary*sRGBPrimaryFactor,false);
threshComparisonSRGB = SRGBGammaCorrect(threshComparisonSRGBPrimary*sRGBPrimaryFactor,false);
 
% Composite an image
nPixels = 512;
blockSize = 110;
blockVertSpacer = 20;
theImage = uint8(ones(nPixels,nPixels,3));
for cc = 1:3
    theImage(:,:,cc) = backgroundSRGB(cc);
end
referenceRowLow = nPixels/2 - blockSize - blockVertSpacer;
referenceRowHigh = referenceRowLow  + blockSize;
referenceColLow = nPixels/2 - blockSize/2;
referenceColHigh = referenceColLow  + blockSize;
for cc = 1:3
    theImage(referenceRowLow:referenceRowHigh,referenceColLow:referenceColHigh,cc) = referenceSRGB(cc);
end
comparisonRowLow = nPixels/2 + blockVertSpacer;
comparisonRowHigh = comparisonRowLow + blockSize;
for tt = 1:length(threshLevels)
    spacer = round((nPixels-length(threshLevels)*blockSize)/(length(threshLevels)+1));
    comparisonColLow = spacer + (tt-1)*blockSize + (tt-1)*spacer;
    comparisonColHigh = comparisonColLow+ blockSize;
    for cc = 1:3
        theImage(comparisonRowLow:comparisonRowHigh,comparisonColLow:comparisonColHigh,cc) = threshComparisonSRGB(cc,tt);
    end
end

% SHow the image
figure; clf; imshow(theImage);
title({'Reference (top) and Comparisons (bottom)' ; ' '});