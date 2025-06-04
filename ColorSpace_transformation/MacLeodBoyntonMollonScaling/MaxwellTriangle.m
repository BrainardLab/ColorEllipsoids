% Maxwell triangle
% 
% Illustrate conversion in and out of a Maxwell triangle.
% Here we use LMS, but any tristimulus coordinates can be
% mapped onto the triangle in this way.
%
% I found this helpful in doing this. Mostly because it defines
% what the numerical coordinates in the triangle are, which
% let me then figure out how to obtain them.
%    https://www.redblobgames.com/x/1730-terrain-shader-experiments/mirror/THE%20PRISMATIC%20COLOR%20SPACE%20FOR%20RGB%20COMPUTATIONS.pdf
% This example, however, led to an isocoles triangle but not an equilateral
% one. I fixed that up in the code below.

% Clear and close
clear; close all hidden;

% Some happy three vectors
theLMS = [ [1 0 0]', [0 1 0]', [0 0 1]', [0.5 0.5 0.5]', [1 1 0]', [1 0 1]', [0 1 1]', ...
    [1 0.5 0.5]', [0.5 1 0.5]', [0.5 0.5 1]', ...
    [2 0.5 0.5]', [0.5 2 0.5]', [0.5 0.5 2]', ...
    [4 0.5 0.5]', [0.5 4 0.5]', [0.5 0.5 4]'];
theColors = theLMS; theColors(theColors > 1) = 1;

% Convert to lms chromaticity using sum of LMS as the normalizer.  You can
% build the triangle with any set of chromaticity coordinates that sum to
% 1, but this version seems to be standard (although done with RGB rather
% than LMS, but that doesn't matter.)
theLMSSum = sum(theLMS,1);
thelms = theLMS ./ theLMSSum;

% The first matrix describes the coordinate system
% on the chromaticities.  Choosing the height this way
% leads to an equalateral triangle, which seems most
% satisfactory.
topVertexHeight = sqrt(1-0.5^2);
M_TriangleToChrom = [[1 0]',[-0.5/topVertexHeight 1/topVertexHeight]'];
M_ChromToTriangle = inv(M_TriangleToChrom);
theTriangle = M_ChromToTriangle*thelms(1:2,:);

% Invert and check that we come back to where we started
thelmCheck = M_TriangleToChrom*theTriangle;
thesCheck = 1-sum(thelmCheck,1);
theLMSCheck = [thelmCheck ; thesCheck] .* theLMSSum(ones(3,1),:);
if (any(abs(thelmCheck-thelms(1:2,:))) > 1e-10)
    error('Cannot self-invert chromaticities')
end
if (any(abs(theLMSCheck-theLMS)) > 1e-10)
    error('Cannot self invert LMS');
end

% Plot the list of colors in the triangle, coloring accoring to
% their LMS coordinates.  Everyone lands where they should
figure; hold on;
for kk = 1:size(theLMS,2)
    plot(theTriangle(1,kk),theTriangle(2,kk),'o','Color',theColors(:,kk) ,'MarkerFaceColor',theColors(:,kk),'MarkerSize',12);
end
plot([0 0.5],[0 topVertexHeight],'k:');
plot([0 1],[0 0],'k:');
plot([0.5 1],[topVertexHeight 0],'k:');
xlim([0 1.25]); 
ylim([0 1.25]);

