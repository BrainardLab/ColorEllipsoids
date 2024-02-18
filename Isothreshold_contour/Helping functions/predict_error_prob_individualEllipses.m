function pIncorrect = predict_error_prob_individualEllipses(ellParam,...
    x_ref, x_comp, alpha_Weibull, beta_Weibull, varargin)
    p = inputParser;
    p.addParameter('visualize',false,@islogical);
    parse(p, varargin{:});
    flag_visualize = p.Results.visualize;

    %x_ref: size of 2 by 1
    %x_comp: size of 2 by N
    majorAxis = 1/ellParam(1);
    minorAxis = 1/ellParam(2);
    rotDeg    = ellParam(3);

    %rotation matrix
    rot_M = [cosd(rotDeg), -sind(rotDeg); sind(rotDeg), cosd(rotDeg)];

    %make them centered around 0
    x_comp_centered = x_comp - x_ref;
    %unrotate them
    x_comp_centered_unrotated = inv(rot_M)*x_comp_centered; 
    %unstretch them
    x_comp_centered_unrotated_unstretched = x_comp_centered_unrotated./[majorAxis;minorAxis]; 

    %now x_comp should be like a unit circle 
    L2_x_comp = sqrt(x_comp_centered_unrotated_unstretched(1,:).^2 +...
        x_comp_centered_unrotated_unstretched(2,:).^2);
    pIncorrect = 1 - ComputeWeibTAFC(L2_x_comp, alpha_Weibull, beta_Weibull);

    %visualize it
    if flag_visualize
        figure
        subplot(2,2,1)
        cmaptemp = colormap('gray'); colormap(flipud(cmaptemp));
        xaxis  = linspace(-0.03,0.03,50) + x_ref(1);
        yaxis  = linspace(-0.03,0.03,50) + x_ref(2);
        h0 = histcounts2(x_comp(1,:), x_comp(2,:), xaxis,yaxis);
        imagesc(xaxis, yaxis, h0); axis square; 
        set(gca,'YDir','normal') 

        subplot(2,2,2)
        xyaxis1 = linspace(-0.03,0.03,50);
        h1 = histcounts2(x_comp_centered(1,:), x_comp_centered(2,:), xyaxis1,xyaxis1);
        imagesc(xyaxis1, xyaxis1, h1); axis square; 
        set(gca,'YDir','normal') 
    
        subplot(2,2,3)
        h2 = histcounts2(x_comp_centered_unrotated(1,:),...
                        x_comp_centered_unrotated(2,:),...
                        xyaxis1,xyaxis1);
        imagesc(xyaxis1, xyaxis1, h2); axis square; 
        set(gca,'YDir','normal')     
    
        subplot(2,2,4)
        xyaxis2 = linspace(-1.6,1.6,50);
        h3 = histcounts2(x_comp_centered_unrotated_unstretched(1,:),...
                        x_comp_centered_unrotated_unstretched(2,:),...
                        xyaxis2,xyaxis2);
        imagesc(xyaxis2, xyaxis2, h3); axis square; 
        set(gca,'YDir','normal')   
    end
end