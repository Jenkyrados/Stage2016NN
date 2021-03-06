function [trainX, trainY, testX, testY] = getMNIST()
    path = fileparts(mfilename('fullpath'));
    if exist('mnist.mat')
        load('mnist.mat');
        return;
    else
        % Download dataset
    if ~exist(fullfile(path, 'train-images-idx3-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 'train-labels-idx1-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 't10k-images-idx3-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 't10k-labels-idx1-ubyte.gz'), 'file')
        fprintf('Downloading dataset...\n');
        websave(fullfile(path, 'train-images-idx3-ubyte.gz'), ...
            'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz');
        gunzip(fullfile(path, 'train-images-idx3-ubyte.gz'));
        websave(fullfile(path, 'train-labels-idx1-ubyte.gz'), ...
            'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz');
        gunzip(fullfile(path, 'train-labels-idx1-ubyte.gz'));

        websave(fullfile(path, 't10k-images-idx3-ubyte.gz'), ...
            'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz');
        gunzip(fullfile(path, 't10k-images-idx3-ubyte.gz'));

        websave(fullfile(path, 't10k-labels-idx1-ubyte.gz'), ...
            'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz');
        gunzip(fullfile(path, 't10k-labels-idx1-ubyte.gz'));

    end
    
    % Load dataset
    fprintf('Extracting dataset...\n');
    
    
    f = fopen(fullfile(path, 'train-images-idx3-ubyte'), 'r', 'b');
    assert(fread(f, 1, 'int32') == 2051, 'magic number verification failed');
    assert(fread(f, 1, 'int32') == 60000, 'expected 60000 samples');
    assert(fread(f, 1, 'int32') == 28);
    assert(fread(f, 1, 'int32') == 28);
    trainX = fread(f, 60000 * 28 * 28, 'uint8=>uint8');
    trainX = reshape(trainX, 28, 28, 60000);
    trainX = permute(trainX, [2, 1, 3]);
    for i = 1:60000
        trainX(:,:,i) = im2double(trainX(:,:,i));
    end
    fclose(f);

    f = fopen(fullfile(path, 'train-labels-idx1-ubyte'), 'r', 'b');
    assert(fread(f, 1, 'int32') == 2049, 'magic number verification failed');
    assert(fread(f, 1, 'int32') == 60000, 'expected 60000 samples');
    trainY = fread(f, 60000, 'uint8=>uint8');
    fclose(f);
    
    f = fopen(fullfile(path, 't10k-images-idx3-ubyte'), 'r', 'b');
    assert(fread(f, 1, 'int32') == 2051, 'magic number verification failed');
    assert(fread(f, 1, 'int32') == 10000, 'expected 60000 samples');
    assert(fread(f, 1, 'int32') == 28);
    assert(fread(f, 1, 'int32') == 28);
    testX = fread(f, 10000 * 28 * 28, 'uint8=>uint8');
    testX = reshape(testX, 28, 28, 10000);
    for i = 1:10000
        testX(:,:,i) = im2double(testX(:,:,i));
    end
    testX = permute(testX, [2, 1, 3]);
    fclose(f);

    f = fopen(fullfile(path, 't10k-labels-idx1-ubyte'), 'r', 'b');
    assert(fread(f, 1, 'int32') == 2049, 'magic number verification failed');
    assert(fread(f, 1, 'int32') == 10000, 'expected 60000 samples');
    testY = fread(f, 10000, 'uint8=>uint8');
    fclose(f);
    save('mnist.mat','trainX','trainY','testX','testY');
    end
end