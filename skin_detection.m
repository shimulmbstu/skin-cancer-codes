
clc
close all

path = 'D:\Research Paper\Skin Cancer\Coding\skin-cancer';

data = fullfile(path, 'skin_cancer'); 
imds = imageDatastore(data, 'LabelSource', 'foldernames', 'IncludeSubfolders',true); 

tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2}); 
maxNumImages = 100;   % change
minSetCount = min(maxNumImages,minSetCount);
imds = splitEachLabel(imds, minSetCount, 'randomize');
countEachLabel(imds)

net = resnet50();
net.Layers(1)

[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');
imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb')
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

featureLayer = 'fc1000';
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% testing image start; loading image

testimage = imageDatastore({'m21.JPG'});
orginalImage = readimage(testimage,1);

% K-mean clustering start

I = im2double(orginalImage);                    % Load Image
F = reshape(I,size(I,1)*size(I,2),3);                 % Color Features
%% K-means
K     = 8;                                            % Cluster Numbers
CENTS = F( ceil(rand(K,1)*size(F,1)) ,:);             % Cluster Centers
DAL   = zeros(size(F,1),K+2);                         % Distances and Labels
KMI   = 10;                                           % K-means Iteration
for n = 1:KMI
   for i = 1:size(F,1)
      for j = 1:K  
        DAL(i,j) = norm(F(i,:) - CENTS(j,:));      
      end
      [Distance, CN] = min(DAL(i,1:K));               % 1:K are Distance from Cluster Centers 1:K 
      DAL(i,K+1) = CN;                                % K+1 is Cluster Label
      DAL(i,K+2) = Distance;                          % K+2 is Minimum Distance
   end
   for i = 1:K
      A = (DAL(:,K+1) == i);                          % Cluster K Points
      CENTS(i,:) = mean(F(A,:));                      % New Cluster Centers
      if sum(isnan(CENTS(:))) ~= 0                    % If CENTS(i,:) Is Nan Then Replace It With Random Point
         NC = find(isnan(CENTS(:,1)) == 1);           % Find Nan Centers
         for Ind = 1:size(NC,1)
         CENTS(NC(Ind),:) = F(randi(size(F,1)),:);
         end
      end
   end
end
X = zeros(size(F));
for i = 1:K
idx = find(DAL(:,K+1) == i);
X(idx,:) = repmat(CENTS(i,:),size(idx,1),1); 
end
T = reshape(X,size(I,1),size(I,2),3);

% k-means clustering end




ds = augmentedImageDatastore(imageSize,orginalImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, ds, featureLayer, 'OutputAs', 'columns');

predictedLabel = predict(classifier,imageFeatures, 'ObservationsIn', 'columns')

% testing image End

figure
imshow(im2bw(T,.6))
title(string(predictedLabel))

%%k flod validation start

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

opt = trainingOptions('sgdm', 'MiniBatchSize', 4 , 'Maxepoch', 15 , 'InitialLearnRate', 0.001, 'Plots','training-progress');
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

