% This file demonstrates the use of the NNBox on the MNIST figure database
% Using the model from Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A
% fast learning algorithm for deep belief nets. Neural computation, 18(7),
% 1527-1554.
tic
nnbox_dir = '../';
addpath(fullfile(nnbox_dir, 'networks'));
addpath(fullfile(nnbox_dir, 'costfun'));
addpath(fullfile(nnbox_dir, 'utils'));


%% Load Database --------------------------------------------------------------

[trainX, trainY, testX, testY] = getMNIST();
trainX = double(reshape(trainX, 28*28, 60000))/255;
trainY = ((0:9)' * ones(1, 60000)) == (ones(10, 1) * double(trainY'));
testX  = double(reshape(testX, 28*28, 10000))/255;
testY  = ((0:9)' * ones(1, 10000)) == (ones(10, 1) * double(testY'));


%% Setup network --------------------------------------------------------------

% Start with an empty multilayer network skeleton
net  = MultiLayerNet();

% Setup first layer
pretrainOpts = struct( ...
    'nEpochs', 50, ...
    'momentum', 0.5, ...
    'momentumChange', 0.9, ...
    'momentumChangeDif', 0.01, ...
    'wDecayDelay',0,...
    'decayNorm',1,...
    'wDecayRate',0.0001,...
    'lRate', 1e-2, ...
    'batchSz', 100, ...
    'dropout', 0.3, ...
    'sparsity', 0.05,...
    'sparseGain', 0.1,...
    'displayEvery', 5);
trainOpts = struct( ...
    'lRate', 5e-4, ...
    'batchSz', 100);
rbm1 = RBM(28*28, 1000, pretrainOpts, trainOpts);

% Add first layer
net.add(rbm1);

% Setup second layer
trainOpts = struct( ...
    'lRate', 5e-4, ...
    'batchSz', 100);
rbm2 = RBM(1000, 1000, pretrainOpts, trainOpts);

% Add second layer
net.add(rbm2);
% rbm5 = RBM(1000, 1000, pretrainOpts, trainOpts);
% 
% % Add second layer
% net.add(rbm5);
% rbm6 = RBM(1000, 1000, pretrainOpts, trainOpts);
% 
% % Add second layer
% net.add(rbm6);
% rbm7 = RBM(1000, 1000, pretrainOpts, trainOpts);
% 
% % Add second layer
% net.add(rbm7);
rbm8 = RBM(1000, 500, pretrainOpts, trainOpts);

% Add second layer
net.add(rbm8);
%% Pretrain network -----------------------------------------------------------
fprintf('Pretraining first two layers\n');
net.pretrain(trainX); % note: MultilayerNet will pretrain layerwise


%% Train ----------------------------------------------------------------------

% Add fully connected layer above
trainOpts = struct( ...
    'lRate', 1e-3, ...
    'batchSz', 100);
rbm3 = RBM(500, 2000, pretrainOpts, trainOpts);
net.add(rbm3);

per  = Perceptron(2000, 10, trainOpts);
net.add(per);

% Train in a supervized fashion
fprintf('Fine-tuning\n');

trainOpts = struct(...
    'nIter', 50, ...
    'batchSz', 500, ...
    'displayEvery', 3);
train(net, CrossEntropyCost(), trainX, trainY, trainOpts);
disp(toc)

%% Results --------------------------------------------------------------------

disp('Confusion matrix:')
[~, tmp] = max(net.compute(testX));
tmp      = tmp - 1; % first class is 0
Y        = bsxfun(@eq, (0:9)' * ones(1, 10000), tmp);
confusion = double(testY) * double(Y');
disp(confusion);

disp('Classification error (testing):');
disp(mean(sum(Y ~= testY) > 0));

disp('Classification error (training):');
[~, tmp] = max(net.compute(trainX));
tmp      = tmp - 1; % first class is 0
Y        = bsxfun(@eq, (0:9)' * ones(1, 60000), tmp);
disp(mean(sum(Y ~= trainY) > 0));

disp('Showing first layer weights as filters (20 largest L2 norm)');
weights = net.nets{1}.W;
[~, order] = sort(sum(weights .^2), 'descend');
colormap gray
for i = 1:20
    subplot(5, 4, i);
    imagesc(reshape(weights(:, order(i)), 28, 28));
    axis image
    axis off
end