% Gets the Faces from sheffield
function [trainX,trainY,validationX, validationY,testX,testY] = getCIFAR()
    if exist('whitenset.mat')
        load('whitenset.mat');
    else
        path = pwd;
        trainX = zeros(40000,3072);
        trainY = [];
        for i = 1:4
            load(fullfile(pwd,'nnbox','examples','cifar-10-batches-mat',strcat('data_batch_',int2str(i),'.mat')));
            for j = 1:10000
                trainX((i-1)*10000+j,:) = whiten(data(j,:),0.0001);
            end
            trainY = vertcat(trainY,labels);
        end
        trainX = trainX';
        load(fullfile(pwd,'nnbox','examples','cifar-10-batches-mat','data_batch_5.mat'));
        validationX = zeros(10000,3072);
        for j = 1:10000
            validationX(j,:) = whiten(data(j,:),0.0001);
        end
        validationX = validationX';
        validationY = labels;
        load(fullfile(pwd,'nnbox','examples','cifar-10-batches-mat','test_batch.mat'));
        testX = zeros(10000,3072);
        for j = 1:10000
            testX(j,:) = whiten(data(j,:),0.0001);
        end
        testX = testX';
        testY = labels;
        save('whitenset.mat','trainX','trainY','validationX','validationY','testX','testY');
    end
end