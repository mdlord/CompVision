%reading all images of the directory
imgDir= 'vision_dataset';
imds = imageDatastore(imgDir,'IncludeSubfolders', true,'LabelSource','foldernames')

tbl = countEachLabel(imds);

mincount = min(tbl{:,2});

imds = splitEachLabel(imds, mincount, 'randomize');

countEachLabel(imds)

[trainingSet, validationSet] = splitEachLabel(imds, 0.3, 'randomize');

% Find the first instance of an image for each category

%airplanes = find(trainingSet.Labels == 'airplanes', 1);
%ferry = find(trainingSet.Labels == 'ferry', 1);
%laptop = find(trainingSet.Labels == 'laptop', 1);

%step = 15;
%bin_size = 8;
%f = [];
features = [];
%pics = dir('*.jpg');
%total_files=length(pics);

K = 100;
dimension = 2;

%loop for the extractiong interest points
%variable SIFT_features stores the descriptors
for i = 1:size(trainingSet.Files)
    
    img = imread(trainingSet.Files{i});
    if (size(img, 3) > 1)
        img =rgb2gray(img);
    end
    
    img = im2single(img);
    
    %lab 
    [f,d] = vl_sift(img);
    
    %[locations, SIFT_features] = vl_dsift(img);
    %features = [features, SIFT_features];
    
    %selects few of the interest points to display
    %perm = randperm(size(f,2)) ;
    %sel = perm(1:100) ;
    
    if(i<5)
        figure(i);
        subplot(1,2,1)
        imshow(img)
        hold on;
        h1 = vl_plotsiftdescriptor(d,f) ;
        hold off;
        
        subplot(1,2,2)
        histogram(d(:,i), 50);
    end
           
        features = [features,im2single(d)]; 

end

% performs clustering
[centers, assignments] = vl_kmeans(double(features), K)
vocab = centers';

vocab_size = size(vocab, 2);

step = 3;
bin_size = 8;
image_feats = [];
forest = vl_kdtreebuild(vocab');

%this loop matches a random point to the nearest cluster point
for i = 1:total_files
    img = single( imread(pics(i).name) );

    if size(img, 3) > 1
        img =rgb2gray(img);
    end
    
    [locations, SIFT_features] = vl_dsift(img, 'step', step, 'size', bin_size);
    
    [index , dist] = vl_kdtreequery(forest , vocab' , double(SIFT_features));
    
    figure;
    feature_hist = hist(double(index), vocab_size);
    feature_hist = feature_hist ./ sum(feature_hist);
    
    image_feats(i, :) = feature_hist;
end


