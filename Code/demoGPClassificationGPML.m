load iris % Loads data matrix named 'data'. Columns 1-4: Sepal.Length Sepal.Width Petal.Length Petal.Width. Col 5 contains the labels 0 = setosa, 1 = versicolor, 2 = virginica

data = data(data(:,5)>0,:);  % using only versicolor and virginica cases, Binary classification.
data(data(:,5)==1,5) = -1; 
data(data(:,5)==2,5) = 1;
y = data(:,5);
%x = data(:,[1 2]);          % using only features Sepal.Length Sepal.Width
x = data(:,[3 4]);          % using only features Petal.Length Petal.Width

figure
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
xlabel('Sepal length','fontsize',12)
ylabel('Sepal width','fontsize',12)
box off
%axis([4 8 2 4])
axis([3 7 1 2.5])
print petalData -depsc2

%[t1 t2] = meshgrid(4:0.1:8,2:0.1:4);
[t1 t2] = meshgrid(3:0.1:7,1:0.1:2.5);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs

disp(' ')
disp('meanfunc = @meanConst; hyp.mean = 0;')
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);')
covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ log(ell_1) log(ell_2) ... log(ell_D) log(sf) ]
disp('likfunc = @likErf;')
likfunc = @likErf;
disp(' ')

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
%hyp.cov(1:2) = [log(2) log(2)]
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
figure
set(gca, 'FontSize', 12)
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
box off
xlabel('Sepal length','fontsize',12)
ylabel('Sepal width','fontsize',12)
%axis([4 8 2 4])
axis([3 7 1 2.5])
title('EP')
print petalMinHypEP -depsc2



%%%%% DATA 1 %%%%%%%%


load data1 % contains x and y
min1 = min(x(:,1));
max1 = max(x(:,1));
min2 = min(x(:,2));
max2 = max(x(:,2));

figure
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
box off
axis([min1 max1 min2 max2])
print DataSet1 -depsc2


[t1 t2] = meshgrid(min1:0.1:max1,min2:0.1:max2);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs

disp(' ')
disp('meanfunc = @meanConst; hyp.mean = 0;')
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);')
covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ log(ell_1) log(ell_2) ... log(ell_D) log(sf) ]
disp('likfunc = @likErf;')
likfunc = @likErf;
disp(' ')

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
%hyp.cov(1:2) = [log(2) log(2)]
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
figure
set(gca, 'FontSize', 12)
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
box off
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
%axis([4 8 2 4])
axis([min1 max1 min2 max2])
title('EP')
print DataSet1MinHypEP -depsc2


%%%%% DATA 2 %%%%%%%%


load data2 % contains x and y
min1 = min(x(:,1));
max1 = max(x(:,1));
min2 = min(x(:,2));
max2 = max(x(:,2));

figure
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
box off
axis([min1 max1 min2 max2])
print DataSet2 -depsc2


[t1 t2] = meshgrid(min1:0.1:max1,min2:0.1:max2);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs

disp(' ')
disp('meanfunc = @meanConst; hyp.mean = 0;')
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);')
covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ log(ell_1) log(ell_2) ... log(ell_D) log(sf) ]
disp('likfunc = @likErf;')
likfunc = @likErf;
disp(' ')

hyp = minimize(hyp, @gp, -40, @infEP, meanfunc, covfunc, likfunc, x, y);
%hyp.cov(1:2) = [log(2) log(2)]
[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
figure
set(gca, 'FontSize', 12)
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
box off
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
%axis([4 8 2 4])
axis([min1 max1 min2 max2])
title('EP')
print DataSet2MinHypEP -depsc2




%%%%% DATA 3 %%%%%%%%


load data3 % contains x and y
min1 = min(x(:,1));
max1 = max(x(:,1));
min2 = min(x(:,2));
max2 = max(x(:,2));

figure
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
box off
axis([min1 max1 min2 max2])
print DataSet3 -depsc2


[t1 t2] = meshgrid(min1:0.1:max1,min2:0.1:max2);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs

disp(' ')
disp('meanfunc = @meanConst; hyp.mean = 0;')
meanfunc = @meanConst; hyp.mean = 0;
disp('covfunc = @covSEard;   hyp.cov = log([1 1 1]);')
covfunc = @covSEard;   hyp.cov = log([1 1 1]);  % hyp = [ log(ell_1) log(ell_2) ... log(ell_D) log(sf) ]
disp('likfunc = @likErf;')
likfunc = @likErf;
disp(' ')

hyp = minimize(hyp, @gp, -40, @infLaplace, meanfunc, covfunc, likfunc, x, y);
%hyp.cov(1:2) = [log(2) log(2)]
[a b c d lp] = gp(hyp, @infLaplace, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
figure
set(gca, 'FontSize', 12)
plot(x(y<0,1), x(y<0,2), 'b+', 'MarkerSize', 8); hold on
plot(x(y>0,1), x(y>0,2), 'ro', 'MarkerSize', 8);
contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
set(h, 'LineWidth', 2)
colorbar
box off
xlabel('x1','fontsize',12)
ylabel('x2','fontsize',12)
%axis([4 8 2 4])
axis([min1 max1 min2 max2])
title('Laplace')
print DataSet3MinHypLaplace -depsc2








