clear all
close all
clc
%% Step 1: Set up image paths, variables

directory = '/Users/mayankdaswani/Documents/MATLAB/vision_dataset/'; 
%, 'anchor','barrel', 'piano', 'cup', 'lamp', 'pizza'
categories = { 'chair','piano','cup','airplanes','lamp','Faces',...
     'cellphone','stop_sign', 'ceiling_fan', 'revolver'};

numpics = 32;        %number of training pics per class

fprintf('Processing image paths and labels for all training and testing data...\n')

numcat = length(categories);
trainpath = cell(numcat * numpics, 1);  %     cell  96x1
testpath  = cell(numcat * numpics, 1);  %     cell  96x1

trainlabel = cell(numcat * numpics, 1); %     cell  96x1
testlabel  = cell(numcat * numpics, 1); %     cell  96x1


for i=1:numcat
   images = dir( fullfile(directory, 'train', categories{i}, '*.jpg'));
   for j=1:numpics
       trainpath{(i-1)*numpics + j} = fullfile(directory, 'train', categories{i}, images(j).name);
       trainlabel{(i-1)*numpics + j} = categories{i};
   end
   
   images = dir( fullfile(directory, 'test', categories{i}, '*.jpg'));
   for j=1:numpics
       testpath{(i-1)*numpics + j} = fullfile(directory, 'test', categories{i}, images(j).name);
       testlabel{(i-1)*numpics + j} = categories{i};
   end
end

% Step 2: Find feature pts

clusterpts = 500; 

fprintf('\nExtracting Features points.. \n')

step = 15;
bin_size = 8;
features = [];

for i = 1:length(trainpath)
    img = single( imread(trainpath{i}) );
     if size(img, 3) > 1
         img =rgb2gray(img);
     end

    img= imresize(img,[100 100]);
    [locations, SIFT_features] = vl_dsift(img, 'fast', 'step', step, 'size', bin_size);
    features = [features, SIFT_features];
    
%     h1 = vl_plotsiftdescriptor(d,f)
    
    
end

[centers, assignments] = vl_kmeans(double(features), clusterpts);
vocab = centers';

save(['vocab_size', num2str(clusterpts),'.mat'], 'vocab')

fprintf('\nfinding Centroids... \n')

train_fts = cluster(trainpath, clusterpts);
save(['train_image_feats_size', num2str(clusterpts), '.mat'], 'train_fts');
          
%%
fprintf('\ncomputing shortest distance... \n')

test_fts  = cluster(testpath, clusterpts);


%% Step 3: Classify each test image by training

fprintf('Using Nearest neighbours to predict test set categories...\n')
fprintf( '\nBuilding Confusion Matrix and Calculating Accuracy.. ')

dist = vl_alldist2(train_fts', test_fts');
dist = dist';


predicted_categories = [];
for i = 1:size(test_fts,1)
    [Y, I] = min(dist(i, :));
    label = trainlabel(I, 1);
%     [idx,d] = knnsearch(SIFT_features,vocab)
    predicted_categories = [predicted_categories; label];
end

numcat = length(categories);

%% Step 4: Create Confusion Matrix

g1 =[];
for i=1:numcat
    for j=1:numpics
        g1 = [g1;categories(i)];
    end

end

confMat = confusionmat(g1,predicted_categories')
confMat = confMat./numpics

accuracy = mean(diag(confMat));
fprintf( '\nAccuracy = %.3f\n', accuracy)


fig_handle = figure; 
imagesc(confMat, [0 1]); 
set(fig_handle, 'Color', [.988, .988, .988])
axis_handle = get(fig_handle, 'CurrentAxes');
set(axis_handle, 'XTick', 1:15)
set(axis_handle, 'XTickLabel', categories)
xlabel('predicted')
set(axis_handle, 'YTick', 1:15)
set(axis_handle, 'YTickLabel', categories)
ylabel('Actual')