params = struct;
params.cf = 'kl';
params.sparsity = 5;
params.max_iter = 100;
params.conv_eps = 1e-3;
params.diplay = 1;
params.random_seed = 1;
params.r = 500;

data = load('imagedata.mat');
v = data.f;
v(1, 1:10)
[W, H, object] = sparse_nmf(v, params);

f = figure();
f.Position = [100 100 1600 400];
subplot(1, 4, 1);
imagesc(v)
subplot(1, 4, 2);
plot(object.div);
hold on
plot(object.cost)
legend(['divergence', 'cost'])
hold off
subplot(1, 4, 3);
imagesc(W)
subplot(1, 4, 4);
imagesc(H)
% savefig('images/eg_matlab.fig')
