clc
close all

path = 'D:\Research Paper\Skin Cancer\Coding\skin-cancer';

data = fullfile(path, 'skin_cancer');
train = imageDatastore(data, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

count = train.countEachLabel;

res=0;

%for k = 1:10
%[imdsTrain, imdsValidation] = splitEachLabel(train, 0.80, 'randomized');

net = alexnet;

layers = [imageInputLayer([256 128 1])
 net(2:end-3)
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer()
];

%layers = setL2Factor(layers ,'Alpha', 2);

opt = trainingOptions('sgdm', 'Maxepoch', 100 , 'InitialLearnRate', 0.001, 'L2Regularization', 1.0000e-04);
training = trainNetwork(train, layers, opt);

[YPred,probs] = classify(training, train);
accuracy = mean(YPred == train.Labels);
accuracy = accuracy*100;

fprintf('Acuracy = %0.2f%%\n', accuracy);

num = numel(train.Labels);
%idx = randperm(num, 50);

TP=0;
TN=0;
FP=0;
FN=0;

%figure
for i = 1:num
    
    %subplot(4,4,i)
    [I, info] = readimage(train, i);
    %imshow(I)
    str1 = string(info.Label);

    out = classify(training, I);
    str2 = string(out);
    
    %ss = strcat(str1, "  ");
    
    %str = strcat(ss, str2);
    
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

fprintf('True Positive(TP) = %0.2f\n', TP);
fprintf('True Negative(TN) = %0.2f\n', TN);
fprintf('False Positive(FP) = %0.2f\n', FP);
fprintf('False Negative(FN) = %0.2f\n', FN);

%end

%res=res/10;

fprintf('Acuracy = %0.2f%%\n', res);

