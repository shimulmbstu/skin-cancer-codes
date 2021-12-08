clc
close all

% K-mean clustering start

orginalImage = imread('m12.jpg');
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

subplot(4,4,1);
imshow(T);title('K mean clustering')
% K-mean clustering end


im=imread('m12.jpg');
fim=mat2gray(im);
level=graythresh(fim);
bwfim=im2bw(fim,0.1);
[bwfim0,level0]=fcmthresh(fim,0);
[bwfim1,level1]=fcmthresh(fim,1);
subplot(4,4,2);
imshow(fim);title('Original');
subplot(4,4,3);
imshow(bwfim);title(sprintf('Otsu,level=%f',level));
subplot(4,4,4);
imshow(bwfim0);title(sprintf('FCM0,level=%f',level0));
subplot(4,4,5);
imshow(bwfim1);title(sprintf('FCM1,level=%f',level1));
% imwrite(bwfim1,'fliver6.jpg');


