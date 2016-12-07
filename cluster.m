function image_feats = cluster(image_paths, vocab_size)

load(['vocab_size', num2str(vocab_size),'.mat']);
vocab_size = size(vocab, 2);

step = 3;
bin_size = 8;
image_feats = [];
forest = vl_kdtreebuild(vocab');

for i = 1:length(image_paths)
    img = single( imread(image_paths{i}) );
    if size(img, 3) > 1
        img =rgb2gray(img);
    end
    img= imresize(img,[100 100]);
    [f,d] = vl_sift(img);
    [locations, SIFT_features] = vl_dsift(img, 'step', step, 'size', bin_size);
    
    [index , dist] = vl_kdtreequery(forest , vocab' , double(SIFT_features));
       
% if ~mod(i,15)
%      figure(i);
%      subplot(1,2,1);
%      imshow(image_paths{i});
%      [f,d] = vl_sift(img);
%      hold on;
%      h1 = vl_plotsiftdescriptor(d,f);
%      hold off;
%      
%      subplot(1,2,2);
%      histogram(double(index), vocab_size);
% end
%     
    feature_hist = hist(double(index), vocab_size);
    feature_hist = feature_hist ./ sum(feature_hist);
    % feature_hist = feature_hist ./ norm(feature_hist);
    
    image_feats(i, :) = feature_hist;
end
