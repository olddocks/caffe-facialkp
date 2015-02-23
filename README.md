# Facial points using Caffe Deep Learning
Facial keypoints extraction using Caffe for kaggle competition

#How to Run
python fkp.py
 ./facialkp
 python output.py

#Description
fkp.py -> to write and prepare all data to hd5
./facialkp -> Run the caffe model
output.py -> Predict and plot graphs in simple 64 batches. it writes into csv
solver.prototxt – > Edit this for maximum iterations, gamma, learning rate etc.
facialkp.prototxt -> Layer file for training
facialkp_predict -> Layer file for predictions
kaggle.py -> writes kaggle output to upload (you have manually edit csv files to add header labels, if not it will not work.

#Requirements
Caffe installed in CUDA enabled GPU
Python/Numpy/Scipy
Scikit-learn and Skimage
Pandas
Ubuntu
HDF5 support in python

#Documentation
 Here: http://corpocrat.com/2015/02/24/facial-keypoints-extraction-using-deep-learning-with-caffe/
