%reading all images of the directory
step = 15;
bin_size = 8;
features = [];
pics = dir('*.jpg');
total_files=length(pics);

K = 100;
dimension = 2;

%loop for the extractiong interest points
%variable SIFT_features stores the descriptors
for i = 1:total_files
    img = single( imread(pics(i).name));
    if size(img, 3) > 1
        img =rgb2gray(img);
    end
    figure(i);
    imshow(pics(i).name)
    [f,d] = vl_sift(I_gray);
    [locations, SIFT_features] = vl_dsift(img, 'fast', 'step', step, 'size', bin_size);
    features = [features, SIFT_features];
    
    %selects few of the interest points to display
    perm = randperm(size(f,2)) ;
    sel = perm(1:100) ;
    h1 = vl_plotsiftdescriptor(d(:,sel),f(:,sel)) ;

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



-
