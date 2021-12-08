clc
close all

path = 'D:\Research Paper\Skin Cancer\Coding';

data = fullfile(path, 'skin-cancer');
train = imageDatastore(data,'IncludeSubfolders',true,'LabelSource','foldernames'); 

count = train.countEachLabel;

%Model name: alexnet, VGG-16, VGG-19, ResNet50; I trained one by one 

net = alexnet;

inputSize = net.Layers(1).InputSize(1:2); 
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);

layers = [imageInputLayer([224 224 3])
 net(2:end-3)
 fullyConnectedLayer(2)
 softmaxLayer
 classificationLayer()
]

opt = trainingOptions('sgdm','Maxepoch',10,'InitialLearnRate',0.0001)
training = trainNetwork(train,layers,opt)


im = imread('D:\Research Paper\Skin Cancer\Coding\skin-cancer\img1.jpg');


out = classify(training, im);

figure,imshow(im)
title(string(out))



