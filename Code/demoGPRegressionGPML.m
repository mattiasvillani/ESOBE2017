%% Simulating data from GP

% Setting up mean in the data generating model
meanfuncData = {@meanSum, {@meanLinear, @meanConst}}; 
hypData.mean = [0.5; 1]; % Linear mean: 1 + 0.5*x

% Setting up the covariance function in the data generating model
covfuncData = {@covSEiso}; 
ell = 1; sf = 0.5; 
hypData.cov = log([ell; sf]); % Squared Exponential, isotropic. length scale is ell. signal variance is sf^2

% Setting up model in the data generating model
likfuncData = @likGauss; 
sn = 0.5; 
hypData.lik = log(sn); % Gaussian likelihood, error variance is sn^2

n = 20; 
x = linspace(-5,5,20)'; % gpml_randn(0.3, n, 1); %
KData = feval(covfuncData{:}, hypData.cov, x);
muData = feval(meanfuncData{:}, hypData.mean, x);
f = muData + chol(KData)'*randn(n,1); % Simulating a draw of f
y = f + exp(hypData.lik)*randn(n,1); % Adding noise: f + error

% Plotting the data set
figure('name','Simulated data')
plot(x,f)
hold on
plot(x,y,'o')
title('Simulated data', 'fontsize',14)
xlabel('x')
ylabel('y')
set(gca,'fontsize',14)
legend('f','data')

%% Estimating a GP

% Setting up mean and covariance functions in the estimated GP
meanfuncEst = {@meanConst}; 
hypEst.mean = [0]; % Zero mean

covfuncEst = {@covSEiso}; 
ell = 1; sf = 0.5; 
hypEst.cov = log([ell; sf]); % Squared Exponential, isotropic. length scale is ell. signal variance is sf^2

likfuncEst = @likGauss; 
sn = 0.5; 
hypEst.lik = log(sn); % Gaussian likelihood, error variance is sn^2


%% Plotting draws from the prior
KEst = feval(covfuncEst{:}, hypEst.cov, x);
muEst = feval(meanfuncEst{:}, hypEst.mean, x);
colors = {'k','b','r','g','m'};
figure('name','Draws from the prior') 
hold on
for i = 1:5
    f = muEst + chol(KEst)'*randn(n,1);
    plot(f,'color',colors{i})
end
title(['Draw from prior with SE kernel and hyperparam l = ',num2str(ell),' and sigma_f = ',num2str(sf)], 'fontsize',14)
xlabel('x')
ylabel('f(x)')
set(gca,'fontsize',14)

%% Fitting the GP using the exact formulas for the GP regression with
% Gaussian noise. Returns the negative log marginal likelihood (lml)
nlml = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y);

% Finding the optimal length scale parameter by gridding the negative lml
nlmls = zeros(10,1);
ells = linspace(0.5,5,10);
count = 0;
for ell = ells
    count = count + 1;
    hypEst.cov(1) = log(ell);
    nlmls(count) = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y);
end
figure('name','Negative Log Marg Like')
plot(ells,nlmls); xlabel('l');ylabel('-LML')
ylims = get(gca,'ylim');
line([exp(hypData.cov(1)) exp(hypData.cov(1))],ylims,'color','r','linestyle','--')

%% Predicting with the estimated model. 
% Note the extra final argument to the gp function.
z = linspace(-5, 5, 100)';
[ymu ys2 fmu fs2] = gp(hypEst, @infExact, meanfuncEst, covfuncEst, likfuncEst, x, y, z);

figure('name','data and predicted curve with posterior error bands') 
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8);
hold on
plot(x,y,'.')

figure('name','Prediction') 
f = [ymu+2*sqrt(ys2); flipdim(ymu-2*sqrt(ys2),1)];
fill([z; flipdim(z,1)], f, [7 7 7]/8, 'EdgeColor','none');
hold on
f = [fmu+2*sqrt(fs2); flipdim(fmu-2*sqrt(fs2),1)];
fill([z; flipdim(z,1)], f, [5 5 5]/8, 'EdgeColor','none');
plot(x,y,'bo')
legend('95% predictive intervals for y','95% predictive intervals for f','data')

hyp2.cov = [0; 0]; hyp2.lik = log(0.1);
hyp2 = minimize(hyp2, @gp, -100, @infExact, [], covfuncEst, likfuncEst, x, y);