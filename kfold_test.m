clc
close all

path = 'D:\Research Paper\Skin Cancer\Coding\skin-cancer';

data = fullfile(path, 'skin_cancer');
train = imageDatastore(data, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

count = train.countEachLabel;

res=0;

%for k = 1:10
[imdsTrain, imdsValidation] = splitEachLabel(train, 0.30, 'randomized');
net = alexnet;

layers = [imageInputLayer([224 224 3])
 net(2:end-3)
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer()
];

opt = trainingOptions('sgdm', 'MiniBatchSize', 4 , 'Maxepoch', 15 , 'InitialLearnRate', 0.001);
training = trainNetwork(imdsTrain, layers, opt);

num = numel(imdsTrain.Labels);
idx = randperm(num, 50);

TP=0;
TN=0;
FP=0;
FN=0;

%figure
for i = 1:50
    
    %subplot(4,4,i)
    [I, info] = readimage(imdsTrain, i);
    imshow(I)
    str1 = string(info.Label);

    out = classify(training, I);
    str2 = string(out);
    
    ss = strcat(str1, "  ");
    
    str = strcat(ss, str2);
    
    %title(str)
    
    if str1 == "melanoma"
        if str2 == "melanoma"
            TP=TP+1;
        else
            FN=FN+1;
        end
        
    else 
        if str2 == "non-melanoma"
            TN=TN+1;
        else
            FP=FP+1;
        end
    end
    
end

res = res + ((TP+TN)/(TP+TN+FP+FN))*100;

%res=res/10;

fprintf('Acuracy = %0.2f%%\n', res);

