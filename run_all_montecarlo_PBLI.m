%% run_all_montecarlo_PBLI.m
% Regenerates the Monte Carlo simulations used for the PBLI validation.
% It saves one .mat file that can be uploaded directly to GitHub/Zenodo.

clear; clc;

rng(1,"twister");

load('pert.mat'); %perturbation vector for pitch-shift UP

cfg.Ntrials       = numel(pert);
cfg.trials        = 1:cfg.Ntrials;
cfg.f0T           = 1;
cfg.thetaTrue     = [0.36 0.45 0.20];      % [alpha_A alpha_S lambda_FF]
cfg.noiseSD_cents = [1 2 4 8 16 32 64 128];
cfg.nMC           = 1000;
cfg.nParticles    = 1000000;
cfg.nPosterior    = 50000;
cfg.startTrialFit = 25;
cfg.credMass      = 0.95;
cfg.restrictions  = "with";
cfg.methods       = ["early-late","late"];

outFile = 'PBLI_montecarlo_results.mat';

blank = initializeBlankResult();
MC_cell = cell(cfg.nMC,1);

if isempty(gcp('nocreate'))
    parpool;
end

parfor mc = 1:cfg.nMC
    localOut = repmat(blank, numel(cfg.noiseSD_cents), numel(cfg.methods));

    for iNoise = 1:numel(cfg.noiseSD_cents)
        noiseSD = cfg.noiseSD_cents(iNoise);

        y = generateSyntheticRealization(cfg.thetaTrue, pert, noiseSD, cfg.f0T);

        R_late  = max(var(1200*log2(y.late(1:24)), 0), 1);
        R_early = max(var(1200*log2(y.early(1:25)),0), 1);

        for iMethod = 1:numel(cfg.methods)
            result = runPBLI(y, pert, cfg, cfg.methods(iMethod), R_late, R_early);

            result.late_produced  = y.late;
            result.early_produced = y.early;
            result.feedforward    = y.feedforward;
            result.noiseSD_cents  = noiseSD;
            result.method         = char(cfg.methods(iMethod));

            localOut(iNoise,iMethod) = result;
        end
    end

    MC_cell{mc} = localOut;
    fprintf('Monte Carlo realization %d/%d\n', mc, cfg.nMC);
end

MC_noise_window = repmat(blank, cfg.nMC, numel(cfg.noiseSD_cents), numel(cfg.methods));

for mc = 1:cfg.nMC
    MC_noise_window(mc,:,:) = MC_cell{mc};
end

MC_noise_window(1,1,1).cfg = cfg;
MC_noise_window(1,1,1).pert = pert;

save(outFile, 'MC_noise_window', 'cfg', '-v7.3');
fprintf('\nSaved: %s\n', outFile);

%% Local functions

function blank = initializeBlankResult()
    blank = struct( ...
        'thetaMAP', [], ...
        'MAP_alpha_A', [], 'MAP_alpha_S', [], 'MAP_lambda_FF', [], ...
        'ETI_alpha_A', [], 'ETI_alpha_S', [], 'ETI_lambda_FF', [], ...
        'HDI_alpha_A', [], 'HDI_alpha_S', [], 'HDI_lambda_FF', [], ...
        'median_alpha_A', [], 'median_alpha_S', [], 'median_lambda_FF', [], ...
        'aA_s', [], 'aS_s', [], 'lF_s', [], ...
        'FF_min', [], 'FF_max', [], 'PR_min', [], 'PR_max', [], ...
        'late_produced', [], 'early_produced', [], 'feedforward', [], ...
        'noiseSD_cents', [], 'method', [], ...
        'cfg', [], 'pert', [] );
end

function y = generateSyntheticRealization(theta, pert, noiseSD_cents, f0T)
    [late0, early0, ff] = simulateSimpleDIVA(theta, pert, f0T);

    eps_cents = noiseSD_cents .* randn(2, numel(pert));
    y.late  = late0  .* 2.^(eps_cents(1,:)/1200);
    y.early = early0 .* 2.^(eps_cents(2,:)/1200);
    y.feedforward = ff;
end

function result = runPBLI(y, pert, cfg, method, R_late, R_early)
    P = drawParticles(cfg.nParticles, pert, cfg.restrictions);
    logw = evaluateParticles(P, y, pert, cfg, method, R_late, R_early);

    logw = logw - max(logw);
    W = exp(logw);
    if ~all(isfinite(W)) || sum(W) <= 0
        W = ones(size(W)) ./ numel(W);
    else
        W = W ./ sum(W);
    end

    [~,idxMAP] = max(W);
    thetaMAP = P(idxMAP,:);

    idxSumm = weightedSampleIdx(W, min(cfg.nPosterior, size(P,1)));
    Ps = P(idxSumm,:);

    result = initializeBlankResult();

    result.thetaMAP = thetaMAP;
    result.MAP_alpha_A   = thetaMAP(1);
    result.MAP_alpha_S   = thetaMAP(2);
    result.MAP_lambda_FF = thetaMAP(3);

    result.aA_s = Ps(:,1);
    result.aS_s = Ps(:,2);
    result.lF_s = Ps(:,3);

    p = [0.025 0.975];
    result.ETI_alpha_A   = wquantile(P(:,1), W, p);
    result.ETI_alpha_S   = wquantile(P(:,2), W, p);
    result.ETI_lambda_FF = wquantile(P(:,3), W, p);

    result.HDI_alpha_A   = whdi_1d(P(:,1), W, cfg.credMass);
    result.HDI_alpha_S   = whdi_1d(P(:,2), W, cfg.credMass);
    result.HDI_lambda_FF = whdi_1d(P(:,3), W, cfg.credMass);

    result.median_alpha_A   = wquantile(P(:,1), W, 0.5);
    result.median_alpha_S   = wquantile(P(:,2), W, 0.5);
    result.median_lambda_FF = wquantile(P(:,3), W, 0.5);

    [result.PR_min, result.PR_max, result.FF_min, result.FF_max] = ...
        credibleTrajectoryEnvelope(P, W, pert, cfg);
end

function P = drawParticles(nParticles, pert, restrictions)
    P = rand(nParticles,3);

    if restrictions == "with"
        Aall = 1 - P(:,3) .* (P(:,1).*(1 + pert) + P(:,2));
        Call = 1 - (P(:,1).*(1 + pert) + P(:,2));
        bad = any(Aall < 0 | Aall > 1, 2) | any(Call < 0 | Call > 1, 2);
        P(bad,:) = [];
    end
end

function logw = evaluateParticles(P, y, pert, cfg, method, R_late, R_early)
    nP = size(P,1);
    logw = zeros(nP,1);
    chunkSize = 25000;
    idxFit = cfg.startTrialFit:cfg.Ntrials;

    yLate_cents  = 1200*log2(y.late(idxFit));
    yEarly_cents = 1200*log2(y.early(idxFit));

    for first = 1:chunkSize:nP
        last = min(first + chunkSize - 1, nP);
        Pc = P(first:last,:);
        [latePred, earlyPred] = simulateMany(Pc, pert, cfg.f0T);

        latePred  = latePred(:,idxFit);
        earlyPred = earlyPred(:,idxFit);

        okLate  = all(latePred  > 0 & isfinite(latePred), 2);
        okEarly = all(earlyPred > 0 & isfinite(earlyPred), 2);

        eLate = 1200*log2(latePred) - yLate_cents;
        lw = -0.5*sum((eLate.^2)./R_late + log(2*pi*R_late), 2);

        if method == "early-late"
            eEarly = 1200*log2(earlyPred) - yEarly_cents;
            lw = lw -0.5*sum((eEarly.^2)./R_early + log(2*pi*R_early), 2);
            ok = okLate & okEarly;
        else
            ok = okLate;
        end

        lw(~ok) = -Inf;
        logw(first:last) = lw;
    end
end

function [late, early, ff] = simulateSimpleDIVA(theta, pert, f0T)
    aA = theta(1); aS = theta(2); lF = theta(3);
    T = numel(pert);

    ff = zeros(1,T+1);
    late = zeros(1,T);
    early = zeros(1,T);
    ff(1) = f0T;

    for k = 1:T
        A = 1 - lF*(aA*(1 + pert(k)) + aS);
        B = lF*(aA + aS);
        C_late = 1 - (aA*(1 + pert(k)) + aS);
        D_late = aA + aS;

        early(k) = ff(k);
        late(k)  = C_late*ff(k) + D_late*f0T;
        ff(k+1)  = A*ff(k) + B*f0T;
    end
end

function [late, early] = simulateMany(P, pert, f0T)
    nP = size(P,1);
    T = numel(pert);

    late = zeros(nP,T);
    early = zeros(nP,T);
    ff = f0T * ones(nP,1);

    aA = P(:,1); aS = P(:,2); lF = P(:,3);

    for k = 1:T
        A = 1 - lF .* (aA.*(1 + pert(k)) + aS);
        B = lF .* (aA + aS);
        C_late = 1 - (aA.*(1 + pert(k)) + aS);
        D_late = aA + aS;

        early(:,k) = ff;
        late(:,k)  = C_late.*ff + D_late*f0T;
        ff = A.*ff + B*f0T;
    end
end

function [PR_min, PR_max, FF_min, FF_max] = credibleTrajectoryEnvelope(P, W, pert, cfg)
    [Ws,idx] = sort(W,'descend');
    cut = find(cumsum(Ws) >= cfg.credMass, 1, 'first');
    Pband = P(idx(1:cut),:);

    PR_min = inf(1,cfg.Ntrials); PR_max = -inf(1,cfg.Ntrials);
    FF_min = inf(1,cfg.Ntrials); FF_max = -inf(1,cfg.Ntrials);

    chunkSize = 25000;
    for first = 1:chunkSize:size(Pband,1)
        last = min(first + chunkSize - 1, size(Pband,1));
        [late, early] = simulateMany(Pband(first:last,:), pert, cfg.f0T);

        PR_min = min(PR_min, min(late,[],1));
        PR_max = max(PR_max, max(late,[],1));
        FF_min = min(FF_min, min(early,[],1));
        FF_max = max(FF_max, max(early,[],1));
    end
end

function idx = weightedSampleIdx(W, n)
    W = W(:) ./ sum(W);
    c = cumsum(W);
    c(end) = 1;
    u = rand(n,1);
    idx = arrayfun(@(x)find(c >= x, 1, 'first'), u);
end

function q = wquantile(x,w,p)
    x = x(:); w = w(:);
    w(~isfinite(w)) = 0;
    if sum(w) <= 0
        w = ones(size(w))/numel(w);
    else
        w = w/sum(w);
    end

    [xs,idx] = sort(x);
    c = cumsum(w(idx));
    [cU,ia] = unique(c,'stable');
    q = interp1(cU, xs(ia), max(0,min(1,p)), 'linear', 'extrap');
end

function hdi = whdi_1d(x,w,credMass)
    x = x(:); w = w(:);
    w(~isfinite(w)) = 0;
    if sum(w) <= 0
        w = ones(size(w))/numel(w);
    else
        w = w/sum(w);
    end

    [xs,idx] = sort(x);
    ws = w(idx);
    cs = cumsum(ws);

    bestWidth = inf;
    bestI = 1; bestJ = numel(xs);
    j = 1;

    for i = 1:numel(xs)
        if j < i
            j = i;
        end

        while j <= numel(xs)
            if i == 1
                mass = cs(j);
            else
                mass = cs(j) - cs(i-1);
            end

            if mass >= credMass
                break
            end
            j = j + 1;
        end

        if j > numel(xs)
            break
        end

        width = xs(j) - xs(i);
        if width < bestWidth
            bestWidth = width;
            bestI = i; bestJ = j;
        end
    end

    hdi = [xs(bestI) xs(bestJ)];
end
