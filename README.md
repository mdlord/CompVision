# Robotics-Computer Vision

3D RECONSTRUCTION (3Dreconstruction.m)consists of the following steps
->Camera Calibration to collect intrinsic parameters(remember to lock the focus AE/AF Lock).
->Feature Matching/ Estimation of Camera Matrices/ Triangulation (Reconstruct the 3D Points in the scene using 2 Frame stereo)
->point cloud formation

BAG OF WORDS (proj3.m )
-> implementation of bag of words using vl_sift/vl-kmeans etc. it uses the code specific funtion cluster.m
->to run this code, you must have vl_feat installed -> follow the link (http://www.vlfeat.org/download.html)
->use seperate files test and train to implement the code(refer to the visio_dataset)

PRETRAINED CONVOLUTION NETWORKS(precnn.m)
->to download the pretrained dataset, follow the link (http://www.vlfeat.org/matconvnet/pretrained/)
->build testing classes based on the labels already in the data set that you are downloading.
