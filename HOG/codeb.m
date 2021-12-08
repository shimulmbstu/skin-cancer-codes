close all;
clear all;
clc;

img=imread('1.jpg');
bw=im2bw(img,0.7);
label=bwlabel(bw);
stats=regionprops(label,'Solidity','Area');
density=[stats.Solidity];
area=[stats.Area];
high_dense_area=density>0.5;
max_area=max(area(high_dense_area));
skin_label=find(area==max_area);
skin=ismember(label,skin_label);
se=strel('square',5);
skin=imdilate(skin,se);
figure(2);
subplot(1,3,1);
imshow(img,[]);
title('Skin Cancer');

subplot(1,3,2);
imshow(skin,[]);
title('Skin Cancer');
[B,L]=bwboundaries(skin,'noholes');
subplot(1,3,3);
imshow(img,[]);
hold on
for i=1:length(B)
    plot(B{i}(:,2),B{i}(:,1), 'y' ,'linewidth',1.45);
end

title('Detected Skin Cancer');
hold off;
    





