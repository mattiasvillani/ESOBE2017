%% GP analysis of Canadian wages data

load CanadianWages
x = (x-mean(x))/std(x); % Standardizing x

% Plotting the data set
figure('name','Simulated data')
plot(x,y,'.')
title('Canadian wages', 'fontsize',14)
xlabel('Age (standardized)','fontsize',14)
ylabel('logWage','fontsize',14)
set(gca,'fontsize',14)

%% Estimating a GP

% Setting up mean and covariance functions in the estimated GP
meanfuncEst = {@meanConst}; hypEst.mean = [0]; % Zero mean
covfuncEst = {@covSEiso}; ell = 1; sf = 10; hypEst.cov = log([ell; sf]); % Squared Exponential, isotropic. length scale is ell. signal variance is sf^2
likfuncEst = @likGauss; sn = std(y); hypEst.lik = log(sn); % Gaussian likelihood, error variance is sn^2

% Fitting the GP using the exact formulas for the GP regression with
% Gaussian noise. Returns the negative log marginal likelihood (lml)
nlml = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y);

% Finding the optimal length scale parameter by gridding the negative lml
nlmls = zeros(100,1);
ells = linspace(0.1,5,100);
count = 0;
for ell = ells
    count = count + 1;
    hypEst.cov(1) = log(ell);
    nlmls(count) = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y);
end
hypEst.cov(1) = ells(find(nlmls==min(nlmls))); % Use the ell value that maximizes the log marginal likelihood
figure('name','Negative Log Marg Like')
plot(ells,nlmls, 'linewidth',2); xlabel('length scale','fontsize',14);ylabel('-LML','fontsize',14)
title(['sigmaf =',num2str(sf)],'fontsize',14)
ylims = get(gca,'ylim');

% Finding the optimal length scale parameter and sigmaf by gridding the negative lml
nlmls = zeros(10,10);
ells = linspace(0.1,5,10);
sigmafs = linspace(60,120,10);
count1 = 0;
count2 = 0;
for sigmaf = sigmafs
    count1 = count1 + 1;
    hypEst.cov(2) = log(sigmaf);
    count2 = 0;
    for ell = ells
        count2 = count2 + 1;
        hypEst.cov(1) = log(ell);
        nlmls(count1,count2) = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y);
    end
end
imagesc(sigmafs,ells,nlmls),colorbar
ylabel('length scale','fontsize',14);
xlabel('sigmaf','fontsize',14)

% Predicting with the estimated model. 
% Note the extra final argument to the gp function.
hypEst.cov(1) = 0.5;
z = linspace(-2, 2.5, 100)';
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y, z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
plot(z,fmu,'r', 'linewidth',2)
plot(x,y,'.')
title('Canadian wages - 95% intervals for f(x)', 'fontsize',14)
xlabel('Age (standardized)','fontsize',14)
ylabel('logWage','fontsize',14)
set(gca,'fontsize',14)
box off

figure('name','Prediction') 
f = [ymu+2*sqrt(ys2); flipdim(ymu-2*sqrt(ys2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [5 5 5]/8, 'EdgeColor','none');
plot(z,fmu,'r', 'linewidth',2)
plot(x,y,'.')
legend('95% predictive intervals for y','95% predictive intervals for f','Predictive mean','data')
title('Canadian wages - 95% intervals for f(x)', 'fontsize',14)
xlabel('Age (standardized)','fontsize',14)
ylabel('logWage','fontsize',14)
set(gca,'fontsize',14)


hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfuncEst, likfuncEst, x, y);

