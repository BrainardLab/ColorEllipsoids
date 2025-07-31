%% Simulate out Bujack ideas
%
% This is basically a simulation of the model in Eq 15 of Bujack et al.
% 2022.  What I'm calling noiseSd is what they call 2*sigma, though.
%
% Here the functions g() and f() of the paper are modeled as simple power
% functions. You get to choose the exponent for each.

% History:
%   07/20/22  dhb  Wrote it.
%   07/21/22  dhb  Allow triples and parameter locking.
%   09/10/22  dhb  Ordered triples as an option, some code cleanup.

%% Clear
clear; close all;
tic;

%% Parameters
%
% Choose a number of stimuli and their range in a physical stimulus space.
nStim = 10;
stimRangeLow = 0;
stimRangeHigh = 1;
stimPhysicalPositions = linspace(stimRangeLow,stimRangeHigh,nStim);

% Choose noiseSd. Here is is omnibus late noise.
noiseSd = 0.2;

% Choose number of trials to simulate per stimulus quadruple
nTrialsPerStim = 1000;

% Set the two non-linearities.  Here they are power functions and and have as
% parameters the exponent. In general these can be defined as functions
% with parameter vectors, but the fitting code below assumes that both pG
% and pF are postive scalars.
funcG = @(x,a,p) (a*(x.^p));
funcF = @(x,a,p) (a*(x.^p));
aG = 1;
pG = 0.5;
aF = 2;
pF = 0.25;

% Search parameters
%
% Specify initial values for each exponent, and
% booleans to lock either at its initial value
initialAg = 1;
initialPg = 1;
initialAf = 1;
initialPf = 1;
lockAg = true;
lockPg = false;
lockAf = false;
lockPf = false;

% Jitter exponents to mimic intermixing of data from different subjects?
jitterExponents = false;
pGJitter = [-0.5 0.5];
pFJitter = [-0.0 0.0];

% Set to true to run all triples instead of quadruples
% Setting ORDERED to true further reduces to ordered triples,
% if TRIPLES is true;
TRIPLES = true;
ORDERED = true;

%% Simulate data for all quadruples, finding simulated pPickFirst.
%
% Various data (stimulis, mapped stimuli, differences) are in each row of
% the first returned matrix. The probability is in the last column.
%
% The second returned matrix has the trial-by-trial simulated data, with
% rows corresponding to those in the first matrix.  A 1 means first
% difference judged larger, a 0 means the second.
if (~jitterExponents)
    [simulatedDataList,simulatedTrialData] = SimulateExperiment(nTrialsPerStim,stimPhysicalPositions, ...
        funcG,aG,pG,funcF,aF,pF,noiseSd,'TRIPLES',TRIPLES,'ORDERED',ORDERED);
else
    [simulatedDataList,simulatedTrialData] = SimulateExperiment(nTrialsPerStim,stimPhysicalPositions, ...
        funcG,aG,pG,funcF,aF,pF,noiseSd,'TRIPLES',TRIPLES, 'ORDERED', ORDERED, ...
        'pGJitter',pGJitter,'pFJitter',pFJitter);
end

%% Get analytic pPickFirst for all quadruples
%
% The same routine computes this when 0 is passed as the number of simulated trials.
% Order of rows matches simulated data above, and probability is in the
% last column.
analyticDataList = SimulateExperiment(0,stimPhysicalPositions,funcG,aG,pG,funcF,aF,pF,noiseSd, ...
    'TRIPLES',TRIPLES,'ORDERED',ORDERED);

% Plot simulated as function of analytic. Converges to identity as
% nTrialsPerDiff gets large. This is basically a check that our simulation
% and calculation of probability match up as expected.
CHECKPLOTS = true;
if (CHECKPLOTS)
    figure; clf;
    subplot(1,2,1); hold on;
    plot(analyticDataList(:,end),simulatedDataList(:,end),'ro','MarkerFaceColor','r','MarkerSize',4);
    plot([0 1],[0 1],'k')
    xlim([0 1]); ylim([0 1]);
    xlabel('Analytic pick first difference (simiulated)');
    ylabel('Simulated pick first difference');
    title(sprintf('%d trials per difference pair',nTrialsPerStim));
    axis('square');
    drawnow;
end

%% Compute log likelihood of the trial by trial data
%
% This is just the trial-by-trial sum of the Bernoulli log likelihoods.
% Pass the simulated trial data and a column vector of the probability tha
% the first difference was judged larger (outcome of 1 in the
% trial-by-trial data).
logLikely = ComputeLogLikelihood(simulatedTrialData,analyticDataList(:,end));

%% Search over the parameters to maximize likelihood of the fitted data
x0 = [initialAg initialPg initialAf initialPf];
vlb = [1e-2 1e-2 1e-2 1e-2];
vub = [1e2 1e2 1e2 1e2];
if (lockAg)
    vlb(1) = initialAg;
    vub(1) = initialAg;
end
if (lockPg)
    vlb(2) = initialPg;
    vub(2) = intiialPg;
end
if (lockAf)
    vlb(3) = initialAf;
    vub(3) = initialAf;
end
if (lockPf)
    vlb(4) = initialPf;
    vub(4) = initialPf;
end

options = optimset('fmincon');
options = optimset(options,'Diagnostics','off','Display','off','LargeScale','off','Algorithm','active-set');
fminconStart=tic;
xFit = fmincon(@(x)FitBujakFunction(x,simulatedTrialData,stimPhysicalPositions,funcG,funcF,noiseSd,TRIPLES,ORDERED),x0,[],[],[],[],vlb,vub,[],options);
fminconTime=toc(fminconStart);

% Report simulated and fit parameters
fprintf('Simulated aG = %0.2f, pG = %0.2f, aF = %0.2f, pF = %0.2f\n', aG, pG, aF, pF);
fprintf('Fit       aG = %0.2f, pG = %0.2f, aF = %0.2f, pF = %0.2f\n', xFit(1), xFit(2), xFit(3), xFit(4));

% Report time to run
toc

analyticFitDataList = SimulateExperiment(0,stimPhysicalPositions,funcG,xFit(1),xFit(2),funcF,xFit(3),xFit(4),noiseSd, ...
    'TRIPLES',TRIPLES,'ORDERED',ORDERED);
if (CHECKPLOTS)
    subplot(1,2,2); hold on;
    plot(analyticFitDataList(:,end),simulatedDataList(:,end),'ro','MarkerFaceColor','r','MarkerSize',4);
    plot([0 1],[0 1],'k')
    xlim([0 1]); ylim([0 1]);
    xlabel('Analytic pick first difference (fit)');
    ylabel('Simulated pick first difference');
    title(sprintf('%d trials per difference pair',nTrialsPerStim));
    axis('square');
    drawnow;
end

function [simulatedDataList,simulatedData] = SimulateExperiment(nTrialsPerStim,stimPhysicalPositions,...
    funcG,aG,pG,funcF,aF,pF,noiseSd,options)
% Simulate out all comparisons of pairwise differences for all possible quadruples
%
% Synopsis:
%    [simulatedDataList,simulatedData] = SimulateExperiment(nTrialsPerStim,stimPhysicalPositions,funcG,aG,pG,funcF,aF,pF,noiseSd)
%
% Description:
%    Go through all possible nStim^4 quadruples and simulate the
%    probability that the first difference is judged larger than the
%    second.  Return info about each quadruple and the simulated
%    probability, as well as the simulated trial-by-trial outcomes.
%
%    Allows for a one parameter mapping between stimuli and perceptual
%    scale, and a one parameter mapping of scale differences to perceptual
%    difference (functions g() and f() of Bujack et al.)  You get to pass
%    a function handle for each and a parameter vector for each, so this routine is
%    pretty general.
%
%    This runs out all possible quadruples, including ones where all
%    stimuli are the same and ones where differences are equal. One could
%    filter the results to reduce to triads, etc.  Indeed, setting
%    key/value TRIPLES to true runs only the nStim^3 triples (stim11 == stim21).
%
%    Setting key/value ORDERED to true limits to ordered triples if TRIPLES
%    is also true, and has no effect for quadruples.
%
% Inputs:
%    nTrialsPerStim             - Scalar. Number of trials to simulate per quadruple.
%    stimPhysicalPositions      - Vector. Physical positions of stimuli
%    funcG                      - Function handle mapping positions to perceptual scale.
%    aG                         - Scalar for funcG
%    pG                         - Exponent for funcG.
%    funcF                      - Function handle mapping scale abs differences to perceptual differences
%    aF                         - Scalar for funcF
%    pF                         - Exponent for funcG.
%    noiseSd                    - SD of zero mean Gaussian noise applied to difference of percptual differences.
%
% Outputs:
%    simulatedDataList          - Matrix with results for each quadruple
%                                   Columns 1-4: stim positions for each member of quadruple
%                                   Columns 5-8: perceptual positions for each
%                                   Columns 9, 10: The two perceptual differences (stim 1 to 2 and 3 to 4)
%                                   Column 11: The signed difference column 9 less column 10
%                                   Column 12: The simulated probability that first diff judged larger than second
%    simulatedData:
%                               - Matrix with trial-by-trial results. 1
%                                 means first difference judged larger, 0 smaller. Row
%                                 order matches that in simulatedDataList.
%
% Optional key/value pairs:
%   'TRIPLES'                   - Boolean. Set to true to compute only for
%                                 triples. Default false.
%   'ORDERED'                   - Boolean. Set to true to compute only for
%                                 ordered triples, if key/value TRIPLES is true.
%   'pGJitter'                  - Jitter amount on pG.  Default empty (no jitter)
%   'pFJitter'                  - Jitter amount on pF. Default empty (no jitter).
%
% See also:
%

% History:
%    07/20/22  dhb Wrote it.

% Parse key value/pairs
arguments
    nTrialsPerStim
    stimPhysicalPositions
    funcG
    aG
    pG
    funcF
    aF
    pF
    noiseSd
    options.TRIPLES = false;
    options.ORDERED = false;
    options.pGJitter = []
    options.pFJitter = []
end

% Loop over all quadruples. This is brute force,
% four nested loops.  I'm sure there is some
% more clever way to do it, but this is easy to read
% and understand.  Simulation is inside inner loop.
nStim = length(stimPhysicalPositions);
listIndex = 1;

if (nTrialsPerStim > 0)
    if options.TRIPLES
        stimulatedData = zeros(nStim^3,nTrialsPerStim);
        simulatedDataList = zeros(nStim^3,12);
    else
        simulatedData = zeros(nStim^4,nTrialsPerStim);
        simulatedDataList = zeros(nStim^4,12);

    end
else
    simulatedData = [];
end

% Loop over first difference
for ii = 1:nStim
    for jj = 1:nStim
        % Pick first difference and compute perceptual representation of
        % difference
        stim11 = stimPhysicalPositions(ii);
        stim12 = stimPhysicalPositions(jj);
        if (~isempty(options.pGJitter))
            pGUse = pG + unifrnd(options.pGJitter(1),options.pGJitter(2));
        else
            pGUse = pG;
        end
        if (pGUse < 0)
            pGUse = 0;
        end
        perc11 = funcG(stim11,aG,pGUse);
        perc12 = funcG(stim12,aG,pGUse);
        if (~isempty(options.pFJitter))
            pFUse = pF + unifrnd(options.pFJitter(1),options.pFJitter(2));
        else
            pFUse = pF;
        end
        if (pFUse < 0)
            pFUse = 0;
        end
        diff1 = funcF(abs(perc11-perc12),aF,pFUse);

        % Loop over second difference for each first difference
        for kk = 1:nStim
            for ll = 1:nStim
                % Pick second difference and compute perceptual
                % representation of difference
                stim21 = stimPhysicalPositions(kk);
                stim22 = stimPhysicalPositions(ll);
                perc21 = funcG(stim21,aG,pGUse);
                perc22 = funcG(stim22,aG,pGUse);
                diff2 = funcF(abs(perc21-perc22),aF,pFUse);

                % Difference of diffs
                diffDiff = diff1-diff2;

                % By setting TRIPLES to true, this has the effect of only
                % considering cases where both stimulus pairs have the same
                % reference.
                if (~options.TRIPLES ...
                        || (options.TRIPLES & ~options.ORDERED & stim11 == stim21) ...
                        || (options.TRIPLES & options.ORDERED & stim11 == stim21 & stim11 > stim12 & stim21 < stim22) )

                    % Simulate out diffs for all trials for this stimulus triple/quadruple.
                    if (nTrialsPerStim > 0)
                        trialVals = normrnd(diffDiff,noiseSd,nTrialsPerStim,1);
                        simulatedData(listIndex,:) = ones(1,nTrialsPerStim);
                        simulatedData(listIndex,trialVals < 0) = 0;
                        pPick1 = sum(simulatedData(listIndex,:))/nTrialsPerStim;

                        % Or if 0 passed as number of trials, compute analytic likelihood of pPick1
                    else
                        pPick1 = 1-normcdf(0,diffDiff,noiseSd);
                        simulatedTrialData = [];
                    end

                    % Store data for this quadruple
                    simulatedDataList(listIndex,1) = stim11;
                    simulatedDataList(listIndex,2) = stim12;
                    simulatedDataList(listIndex,3) = stim21;
                    simulatedDataList(listIndex,4) = stim22;
                    simulatedDataList(listIndex,5) = perc11;
                    simulatedDataList(listIndex,6) = perc12;
                    simulatedDataList(listIndex,7) = perc21;
                    simulatedDataList(listIndex,8) = perc22;
                    simulatedDataList(listIndex,9) = diff1;
                    simulatedDataList(listIndex,10) = diff2;
                    simulatedDataList(listIndex,11) = diffDiff;
                    simulatedDataList(listIndex,12) = pPick1;
                    listIndex = listIndex+1;
                end
            end
        end
    end
end

% Check that we allocated space right.  Haven't yet decided on right answer
% for ordered triples, so don't check in that case.  But do truncate so we
% don't have extra stuff.
if (options.TRIPLES & ~options.ORDERED)
    if (listIndex-1 ~= nStim^3)
        error('Something wrong in pre-allocation logic');
    else
        if (nTrialsPerStim > 0)
            fprintf('Simulating for %d unordered triples\n',listIndex-1);
        end
    end
elseif (options.TRIPLES & options.ORDERED)
    if (any(simulatedData(listIndex:end,:) ~= 0))
        error('Do not understand how simulatedData are being filled in');
    end
    simulatedDataList = simulatedDataList(1:listIndex-1,:);
    if (nTrialsPerStim > 0)
        fprintf('Simulating for %d ordered triples\n',listIndex-1);
    end
elseif (~options.TRIPLES)
    if (listIndex-1 ~= nStim^4)
        error('Something wrong in pre-allocation logic');
    else
        if (nTrialsPerStim > 0)
            fprintf('Simulating for %d unordered quadruples\n',listIndex-1);
        end
    end
end

end

function logLikely = ComputeLogLikelihood(simulatedTrialData,probs)
% Compute likelihood of the data.
%
% Synopsis:
%    logLikely = ComputeLogLikelihood(simulatedTrialData,probs)
%
% Description:
%    For any stimulus quadruple, trials are Bernoulli tnd we
%    have the probability for each, so this is easy.
%
%    Converts passed probs less than a threshold to that threshold, and
%    more than 1-thresh to 1-thresh. This avoids pathalogical log
%    likelihoods. The threshold is currently set in the code.  Could be
%    handled as a key/value pair.
%
% Inputs:
%    simulatedTrialData    - Simulated trial data as returned by
%                            SimulateExperiment
%    probs                 - Column vector with probabilites for picking
%                            first as larger. Should correspond with rows
%                            of simulatedTrialData.
%
% Outputs:
%   logLikely              - Log10 likelihood of data given probs
%
% Optional key/value pairs:
%   None.
%
% See also: SimulateExperiment

% History
%  07/20/22  dhb  Wrote it.

% Avoid pathological cases
thresh = 1e-4;
probs(probs > 1-thresh) = 1-thresh;
probs(probs < thresh) = thresh;

% Loop over all trials and sum up log likelihood.
logLikely = 0;
for ll = 1:size(simulatedTrialData,1)
    for tt = 1:size(simulatedTrialData,2)
        if simulatedTrialData(ll,tt) == 1
            logLikely = logLikely + log10(probs(ll));
        else
            logLikely = logLikely + log10(1-probs(ll));
        end
    end
end
end

function negLogLikely = FitBujakFunction(x,simulatedTrialData,stimPhysicalPositions,funcG,funcF,noiseSd,TRIPLES,ORDERED)
% Error function for parameter fit
%
% Synopsis:
%     negLogLikely = FitBujakFunction(x,simulatedTrialData,stimPhysicalPositions,funcG,funcF,noiseSd,TRIPLES)
%
% Description:
%     Return negative log likelihood of the data given the parameters being
%     searched over.  This is minimized to find maximum likelihood
%     solution.
%
% Inputs:
%     x                          - Parameter vector from fmincon
%     simulatedTrialData         - As returned by SimulateExperiment
%     stimPhysicalPositions      - Vector of stimulus values
%     funcG                      - Handle to g() function
%     funcF                      - Handle to f() function
%     noiseSd                    - Standard deviation of noise
%     TRIPLES                    - Set to true to compute only for triples.
%
% Optional key/value pairs:
%     None
%
% See also: SimulateExperiment, ComputeLogLikelihood

% History:
%   07/20/22  dhb  Wrote it.

% Pull out params by name
aG = x(1);
pG = x(2);
aF = x(3);
pF = x(4);

% Get probabilities for each quadruple
analyticDataList = SimulateExperiment(0,stimPhysicalPositions,funcG,aG,pG,funcF,aF,pF,noiseSd, ...
    'TRIPLES',TRIPLES,'ORDERED',ORDERED);

% Compute negative log likelihood (which is then minimized).
negLogLikely = -ComputeLogLikelihood(simulatedTrialData,analyticDataList(:,end));

% Handle pathology.  Ideally this never happens
if (isinf(negLogLikely) | isnan(negLogLikely))
    negLogLikely = 1e100;
end

end
