function LMS = MacBoynToLMS(ls,T_cones,T_lum, sumLM) 
    %in this version, sumLM = L + M

    factorsLM = (T_cones(1:2,:)'\T_lum');
    factorL = factorsLM(1);
    factorM = factorsLM(2);
    factorS = 1/max(T_cones(3,:)./T_lum);

    l = ls(1);
    s = ls(2);
    denom = l/factorL + (1-l)/factorM;
    L = (l*sumLM / factorL) / denom;
    M = ((1-l)*sumLM/ factorM) / denom;
    S = (s*sumLM / factorS) / denom;

    LMS = [L; M; S];
end