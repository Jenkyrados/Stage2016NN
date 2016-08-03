% This file demonstrates the use of the NNBox on the MNIST figure database
% Using the model from Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). A
% fast learning algorithm for deep belief nets. Neural computation, 18(7),
% 1527-1554.
N = 20;
patchsize = 15;
nnbox_dir = '../';
addpath(fullfile(nnbox_dir, 'networks'));
addpath(fullfile(nnbox_dir, 'costfun'));
addpath(fullfile(nnbox_dir, 'utils'));


%% Load Database --------------------------------------------------------------

[trainX, trainY, testX, testY] = getMNIST();
trainX = double(reshape(trainX, 28,28, 60000));
trainY = ((0:9)' * ones(1, 60000)) == (ones(10, 1) * double(trainY'));
testX  = double(reshape(testX, 28,28, 10000));
testY  = ((0:9)' * ones(1, 10000)) == (ones(10, 1) * double(testY'));


%% Setup network --------------------------------------------------------------
tic

% Start with an empty multilayer network skeleton
net  = MultiLayerNet();
patchMaker = PatchNet([28 28],[patchsize patchsize], [patchsize-1 patchsize-1]);

% Setup first layer
pretrainOpts = struct( ...
    'nEpochs', N, ...
    'dropoutVis',0.2,...
    'dropoutHid',0.5,...
    'momentum', 0.9, ...
    'momentumChange', 0.9, ...
    'momentumChangeDif', 0.01, ...
    'wDecayDelay',0,...
    'decayNorm',2,...
    'decayRate',0.001,...
    'lRate', 1e-1, ...
    'lRateDecay', 0.9,...
    'batchSz', 100, ... 
    'displayEvery', 5);
trainOpts = struct(...
    'lRate', 7e-3, ...
    'batchSz', 100);
rbm1 = RBM(patchsize*patchsize, 500, pretrainOpts, trainOpts);

% Add first layer
net.add(rbm1);

 pretrainOpts.dropoutVis = 0.2;
 % Setup second layer
 trainOpts.dropout = 0.5;
rbm2 = RBM(500, 300, pretrainOpts, trainOpts);
rbm3 = RBM(300, 10, pretrainOpts, trainOpts);

 net.add(rbm2);
 net.add(rbm3);
% rbm3 = RBM(320, 160, pretrainOpts, trainOpts);
% 
% net.add(rbm3);
%% Pretrain network -----------------------------------------------------------
 for i = 1:10
    disp(i);
    fulltrainX = cell(196,1000);
    ex = randperm(60000,1000);
    for j = 1:1000
        fulltrainX(:,j) = patchMaker.compute(trainX(:,:,ex(j)));
    end
    trainingX = reshape(cell2mat(fulltrainX),patchsize*patchsize,196*1000);
    net.pretrain(trainingX,0); % note: MultilayerNet will pretrain layerwise
end
%% Train ----------------------------------------------------------------------
siamese = MultiLayerNet();
siamese.add(SiameseNet(net,(28-patchsize+1)*(28-patchsize+1)));

%Add fully connected layer above
per  = Perceptron((28-patchsize+1)*(28-patchsize+1)*10, 10, trainOpts);
siamese.add(per);

% Train in a supervized fashion
fprintf('Fine-tuning\n');

trainOpts = struct(...
    'nIter', 10, ...
    'batchSz', 100, ...
    'displayEvery', 3);

 for i = 1:10
    disp(i);
    fulltrainX = cell(196,1000);
    fulltrainY = zeros(10,1000);        
    ex = randperm(60000,1000);
    for j = 1:1000
        fulltrainX(:,j) = patchMaker.compute(trainX(:,:,ex(j)));
        fulltrainY(:,j) = trainY(:,ex(j));
    end
    fulltrainY = logical(fulltrainY);
    train(siamese, CrossEntropyCost(), fulltrainX, fulltrainY, trainOpts);
end
toc

%% Results --------------------------------------------------------------------
fulltestX = cell(196,10000);
fulltestY = testY;

for ex = 1:10000;
    fulltestX(:,ex) = patchMaker.compute(testX(:,:,ex));
end
disp('Confusion matrix:')
[~, tmp] = max(siamese.compute(fulltestX));
tmp      = tmp - 1; % first class is 0
Y        = bsxfun(@eq, (0:9)' * ones(1, 10000), tmp);
confusion = double(fulltestY) * double(Y');
disp(confusion);

disp('Classification error (testing):');
disp(mean(sum(Y ~= fulltestY) > 0));

disp('Classification error (training):');
[~, tmp] = max(siamese.compute(fulltrainX));
tmp      = tmp - 1; % first class is 0
Y        = bsxfun(@eq, (0:9)' * ones(1, 1000), tmp);
disp(mean(sum(Y ~= fulltrainY) > 0));

disp('Displaying mean reconstruction error (testing)')
disp(net.nets{1}.recErr(fulltestX));

disp('Displaying mean reconstruction error (training)')
disp(net.nets{1}.recErr(fulltrainX));

disp('Showing first layer weights as filters (20 largest L2 norm)');
weights = net.nets{1}.W;
[~, order] = sort(sum(weights .^2), 'descend');
colormap gray
for i = 1:20
    subplot(5, 4, i);
    imagesc(reshape(weights(:, order(i)), patchsize, patchsize));
    axis image
    axis off
end