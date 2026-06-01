%% run_example_PBLI.m
% Minimal example of the PBLI workflow for one synthetic realization.
% It keeps the same idea as the figures used in the paper: one noisy
% trajectory, MAP fits, and credible envelopes for both likelihoods.

clear; clc; close all;

rng(1,"twister");

load('pert.mat'); %perturbation vector for pitch-shift UP

cfg.Ntrials       = numel(pert);
cfg.trials        = 1:cfg.Ntrials;
cfg.f0T           = 1;
cfg.thetaTrue     = [0.36 0.45 0.20];     % [alpha_A alpha_S lambda_FF]
cfg.noiseSD_cents = [2 8 32];             % only for the example figure
cfg.nParticles    = 1000000;
cfg.nPosterior    = 30000;
cfg.startTrialFit = 25;
cfg.credMass      = 0.95;
cfg.restrictions  = "with";   % 0<A[k]<1 and 0<C[k]<1

for ii = 1:numel(cfg.noiseSD_cents)
    noiseSD = cfg.noiseSD_cents(ii);

    y = generateSyntheticRealization(cfg.thetaTrue, pert, noiseSD, cfg.f0T);

    R_late  = max(var(1200*log2(y.late(1:24)), 0), 1);
    R_early = max(var(1200*log2(y.early(1:25)),0), 1);

    out(ii).noiseSD = noiseSD;
    out(ii).y = y;
    out(ii).dualWindow = runPBLI(y, pert, cfg, "early-late", R_late, R_early);
    out(ii).lateWindow = runPBLI(y, pert, cfg, "late",       R_late, R_early);
end

disp('Ground-truth parameters: [alpha_A alpha_S lambda_FF]')
disp(cfg.thetaTrue)
disp(' ')
disp('MAP estimates')
for ii = 1:numel(out)
    fprintf('\nNoise SD = %g cents\n', out(ii).noiseSD);
    fprintf('  dual-window: [%0.4f  %0.4f  %0.4f]\n', out(ii).dualWindow.thetaMAP);
    fprintf('  late-window: [%0.4f  %0.4f  %0.4f]\n', out(ii).lateWindow.thetaMAP);
end

plotTrajectoryExample(out, pert, cfg);
plotPosteriorExample(out(end), cfg);

%% Local functions

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

function plotTrajectoryExample(out, pert, cfg)
    Colors = lines(7);
    trials = cfg.trials;
    idx = cfg.startTrialFit:cfg.Ntrials;

    [lateTrue, earlyTrue] = simulateSimpleDIVA(cfg.thetaTrue, pert, cfg.f0T);

    figure('Color','w','Name','PBLI trajectory example');
    tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

    for ii = 1:numel(out)
        y = out(ii).y;
        dual = out(ii).dualWindow;
        lateOnly = out(ii).lateWindow;

        [lateMAP_dual, earlyMAP_dual] = simulateSimpleDIVA(dual.thetaMAP, pert, cfg.f0T);
        [lateMAP_late, earlyMAP_late] = simulateSimpleDIVA(lateOnly.thetaMAP, pert, cfg.f0T);

        nexttile(ii); hold on; box on; grid on;

        hLateBand = fill([trials(idx) fliplr(trials(idx))], ...
            [lateOnly.FF_min(idx) fliplr(lateOnly.FF_max(idx))], ...
            Colors(2,:), 'FaceAlpha',0.18, 'EdgeColor','none');

        hDualBand = fill([trials(idx) fliplr(trials(idx))], ...
            [dual.FF_min(idx) fliplr(dual.FF_max(idx))], ...
            Colors(1,:), 'FaceAlpha',0.30, 'EdgeColor','none');

        hData = plot(trials, y.early, 'k-', 'LineWidth',0.8);
        hTrue = plot(trials, earlyTrue, '-', 'Color',Colors(5,:), 'LineWidth',1.2);
        hDual = plot(trials(idx), earlyMAP_dual(idx), '--', 'Color',Colors(1,:), 'LineWidth',2.2);
        hLate = plot(trials(idx), earlyMAP_late(idx), ':', 'Color',Colors(2,:), 'LineWidth',2.4);

        ylim([0.965 1.015]);
        xlim([1 cfg.Ntrials]);
        title(sprintf('Noise SD = %g cents', out(ii).noiseSD));
        if ii == 1
            ylabel('Early-window f_o (norm.)');
        end

        yyaxis right
        ylim([0.965 1.015]);
        yticks(2.^((-100:20:100)/1200));
        yticklabels({'-100','-80','-60','-40','-20','0','+20','+40','+60','+80','+100'});
        if ii == 3
            ylabel('cents');
        end
        yyaxis left

        if ii == 1
            legend([hData hTrue hDual hDualBand hLate hLateBand], ...
                {'Simulated noisy data','Noise-free trajectory', ...
                 'MAP (dual-window)','95% envelope (dual-window)', ...
                 'MAP (late-window)','95% envelope (late-window)'}, ...
                 'Location','southwest','Box','off');
        end

        nexttile(ii+3); hold on; box on; grid on;

        hLateBand = fill([trials(idx) fliplr(trials(idx))], ...
            [lateOnly.PR_min(idx) fliplr(lateOnly.PR_max(idx))], ...
            Colors(2,:), 'FaceAlpha',0.18, 'EdgeColor','none');

        hDualBand = fill([trials(idx) fliplr(trials(idx))], ...
            [dual.PR_min(idx) fliplr(dual.PR_max(idx))], ...
            Colors(1,:), 'FaceAlpha',0.30, 'EdgeColor','none');

        plot(trials, y.late, 'k-', 'LineWidth',0.8);
        plot(trials, lateTrue, '-', 'Color',Colors(5,:), 'LineWidth',1.2);
        plot(trials(idx), lateMAP_dual(idx), '--', 'Color',Colors(1,:), 'LineWidth',2.2);
        plot(trials(idx), lateMAP_late(idx), ':', 'Color',Colors(2,:), 'LineWidth',2.4);

        ylim([0.965 1.015]);
        xlim([1 cfg.Ntrials]);
        xlabel('Trial');
        if ii == 1
            ylabel('Late-window f_o (norm.)');
        end

        yyaxis right
        ylim([0.965 1.015]);
        yticks(2.^((-100:20:100)/1200));
        yticklabels({'-100','-80','-60','-40','-20','0','+20','+40','+60','+80','+100'});
        if ii == 3
            ylabel('cents');
        end
        yyaxis left
    end
end

function plotPosteriorExample(exampleOut, cfg)
    Colors = lines(7);
    dual = exampleOut.dualWindow;
    lateOnly = exampleOut.lateWindow;

    figure('Color','w','Name','PBLI posterior example');
    tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

    names = {'\alpha_A','\alpha_S','\lambda_{FF}'};
    trueVals = cfg.thetaTrue;
    dualSamples = {dual.aA_s, dual.aS_s, dual.lF_s};
    lateSamples = {lateOnly.aA_s, lateOnly.aS_s, lateOnly.lF_s};
    dualHDI = {dual.HDI_alpha_A, dual.HDI_alpha_S, dual.HDI_lambda_FF};
    lateHDI = {lateOnly.HDI_alpha_A, lateOnly.HDI_alpha_S, lateOnly.HDI_lambda_FF};
    dualMAP = dual.thetaMAP;
    lateMAP = lateOnly.thetaMAP;

    for p = 1:3
        nexttile(2*p-1); hold on; box on;
        histogram(dualSamples{p}, 80, 'Normalization','pdf', 'FaceColor',Colors(1,:), 'EdgeColor','none', 'FaceAlpha',0.55);
        xline(trueVals(p),'k-','LineWidth',1.4);
        xline(dualMAP(p),'--','Color',Colors(1,:), 'LineWidth',1.8);
        xline(dualHDI{p},'-','Color',Colors(4,:), 'LineWidth',1.5);
        xlim([0 1]);
        ylabel('Density');
        title(['Dual-window: ', names{p}], 'Interpreter','tex');

        if p == 1
            legend({'Posterior samples','True value','MAP','95% HDI'}, ...
                'Box','off','Location','best');
        end

        nexttile(2*p); hold on; box on;
        histogram(lateSamples{p}, 80, 'Normalization','pdf', 'FaceColor',Colors(2,:), 'EdgeColor','none', 'FaceAlpha',0.55);
        xline(trueVals(p),'k-','LineWidth',1.4);
        xline(lateMAP(p),':','Color',Colors(2,:), 'LineWidth',2.2);
        xline(lateHDI{p},'-','Color',Colors(4,:), 'LineWidth',1.5);
        xlim([0 1]);
        title(['Late-window: ', names{p}], 'Interpreter','tex');
    end
end
