% Gets the Faces from sheffield
function [trainX,trainY,testX,testY] = getFACES()
    path = fileparts(mfilename('fullpath'));
    if ~exist(fullfile(path, 'cropped.tar.gz'), 'file')
         websave(fullfile(path, 'cropped.tar.gz'), ...
            'http://eeepro.shef.ac.uk/vie/cropped.tar.gz');
        untar(fullfile(path,'cropped.tar.gz'),fullfile(path,'cropped'));
        delete(fullfile(path,'cropped.tar.gz'));
    end
    total=[];
    labels=[];
    i = 0;
    folders = dir(fullfile(path,'cropped'));
    folders = folders([folders.isdir]);
    folders = folders(arrayfun(@(x) x.name(1), folders) ~= '.');
    for folder = folders'
        foldersbis = dir(fullfile(path,'cropped',folder.name));
        foldersbis = foldersbis([foldersbis.isdir]);
        foldersbis = foldersbis(arrayfun(@(x) x.name(1), foldersbis) ~= '.');
        for folderbis = foldersbis'
            ims = dir(fullfile(path,'cropped',folder.name,folderbis.name));
            ims = ims(arrayfun(@(x) x.name(1), ims) ~= '.');
            for im = ims'
                labels = [labels i];
                train = imread(fullfile(path,'cropped',folder.name,folderbis.name,im.name));
                total = [total train];
            end
        end
        i = i+1;
    end
    total = double(reshape(total, 112*92, 575))/255;
    idx = randperm(575);
    trainX = total(:,idx(1:floor(575/2)));
    testX = total(:,idx(floor(575/2)+1:end));
    trainY = labels(idx(1:floor(575/2)));
    trainY = ((0:19)' * ones(1, floor(575/2))) == (ones(20, 1) * double(trainY));
    testY = labels(idx(floor(575/2)+1:end));
    testY = ((0:19)' * ones(1, floor(575/2)+1)) == (ones(20, 1) * double(testY));

end