% Gets the Faces from sheffield
function trainX = getFACES()
    path = fileparts(mfilename('fullpath'));
    if ~exist(fullfile(path, 'cropped.tar.gz'), 'file')
         websave(fullfile(path, 'cropped.tar.gz'), ...
            'http://eeepro.shef.ac.uk/vie/cropped.tar.gz');
        untar('cropped.tar.gz');
    end
    trainX = [];
    folders = dir(fullfile(path,'cropped'));
    folders = folders([folders.isdir]);
    for folder = folders'
        foldersbis = dir(fullfile(path,'cropped',folder.name));
        foldersbis = foldersbis([foldersbis.isdir]);
        for folderbis = foldersbis'
            for im = dir(fullfile(path,'cropped',folder.name,folderbis.name));
                train = imread(im.name);
                trainX = [trainX train];
            end
        end
    end
end