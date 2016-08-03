function [trainX, trainY, testX, testY] = getNORB(randamount , size)
    path = fileparts(mfilename('fullpath'));
    
    % Download dataset
    if ~exist(fullfile(path, 'train-images-idx3-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 'train-labels-idx1-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 't10k-images-idx3-ubyte.gz'), 'file') ...
        || ~exist(fullfile(path, 't10k-labels-idx1-ubyte.gz'), 'file')
        fprintf('Downloading dataset...\n');
        websave(fullfile(path, 'train-images-idx3-ubyte.gz'), ...
            'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz');
        websave(fullfile(path, 'train-labels-idx1-ubyte.gz'), ...
            'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz');
        websave(fullfile(path, 't10k-images-idx3-ubyte.gz'), ...
            'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz');
        websave(fullfile(path, 't10k-labels-idx1-ubyte.gz'), ...
            'http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz');
        
    end
    
    % Load dataset
    fprintf('Extracting dataset...\n');
    
    gunzip(fullfile(path, 'train-images-idx3-ubyte.gz'));
    f = fopen(fullfile(path, 'train-images-idx3-ubyte'), 'r', 'b');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    trainX = fread(f, 48600 * 96 * 96,'uint8=>uint8');
    trainX = swapbytes(trainX);
    trainX = reshape(trainX, 96, 96, 48600);
    if nargin > 1
        realX = zeros(size, size, randamount);
        for i = 1:randamount
            realX(:,:,i) = trainX(randi(96-size+1)+(0:size-1), ...
                randi(96-size+1)+(0:size-1), randi(48600));
        end
        trainX = realX;
    end
    trainX = permute(trainX, [2, 1, 3]);
    fclose(f);
    delete(fullfile(path, 'train-images-idx3-ubyte'));

    gunzip(fullfile(path, 'train-labels-idx1-ubyte.gz'));
    f = fopen(fullfile(path, 'train-labels-idx1-ubyte'), 'r', 'b');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    trainY = uint8(fread(f, 24300, 'uint8'));
    trainY = swapbytes(trainY);
    trainY = repmat(trainY,1,2)';
    trainY = trainY(:);
    fclose(f);
    delete(fullfile(path, 'train-labels-idx1-ubyte'));
    
    gunzip(fullfile(path, 't10k-images-idx3-ubyte.gz'));
    f = fopen(fullfile(path, 't10k-images-idx3-ubyte'), 'r', 'b');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    testX = fread(f, 48600 * 96 * 96);
    testX = swapbytes(testX);
    testX = reshape(testX, 96, 96, 48600);
    testX = permute(testX, [2, 1, 3]);
    fclose(f);
    delete(fullfile(path, 't10k-images-idx3-ubyte'));

    gunzip(fullfile(path, 't10k-labels-idx1-ubyte.gz'));
    f = fopen(fullfile(path, 't10k-labels-idx1-ubyte'), 'r', 'b');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    fread(f, 4, 'uchar');
    testY = uint8(fread(f, 24300, 'uint8'));
    testY = swapbytes(testY);
    testY = repmat(testY,1,2)';
    testY = testY(:);
    fclose(f);
    delete(fullfile(path, 't10k-labels-idx1-ubyte'));
end