# Kaggle Facial points detection using Caffe Deep Learning
Facial keypoints extraction using Caffe for kaggle competition https://www.kaggle.com/c/facial-keypoints-detection
This problem is a classic multilabel regression problem to solve. The kaggle CSV file provides (96,96) pixel images and you have to predict 30 keypoints (x,y) coordinates of nose, eye_center etc. The challenge ataset is over 70% of the data is missing filled with NaNs.

#Description of Files
```
fkp.py -> to write and prepare all data to hd5
./facialkp -> Run the caffe model
output.py -> Predict and plot graphs in simple 64 batches. it writes into csv
solver.prototxt – > Edit this for maximum iterations, gamma, learning rate etc.
facialkp.prototxt -> Layer file for training
facialkp_predict -> Layer file for predictions
kaggle.py -> writes kaggle output to upload (you have manually edit csv files to add header labels, if not it will not work.
```
# How to run
```
python fkp.py //run to preapare all data
./facialkp.sh //run the caffe trainer
python output.py // predicts the results and dumps the results in csv
python kaggle.py // writes the kaggle output to kaggle.csv 
```
#Requirements
```
Caffe installed in CUDA enabled GPU
Python/Numpy/Scipy
Scikit-learn and Skimage
Pandas
Ubuntu
HDF5 support in python
```
#Documentation
 Here: http://corpocrat.com/2015/02/24/facial-keypoints-extraction-using-deep-learning-with-caffe/
