import numpy as np
import pandas as pd
from numpy import genfromtxt
from numpy import ravel
import pylab as pl
from skimage import transform
import h5py
from sklearn import cross_validation
import uuid
import random
from skimage import io, exposure, img_as_uint, img_as_float
from numpy import (array, dot, arccos)
from numpy.linalg import norm

df = pd.read_csv('training.csv',header=0)
dfp = pd.read_csv('test.csv',header=0)

#calculate the distance features

#df = df.interpolate()
#df = df.reindex(np.random.permutation(df.index))

def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf



df = df.dropna()
y = df.drop(['Image'], axis=1)
#y = y.interpolate()
y = y.values 
y = y.astype(np.float32) 
y = y.reshape((-1,30))

y_features = np.zeros([len(y),15], np.float32)

for k in xrange(len(y)):

 mouth_right_x = y[k,24]
 mouth_right_y = y[k,25]
 mouth_left_x =  y[k,22]
 mouth_left_y = y[k,23]
 mouth_top_x =  y[k,26]
 mouth_top_y =  y[k,27]
 mouth_bottom_x = y[k,28]
 mouth_bottom_y = y[k,29]
 
 left_eye_brow_left_x = y[k,14]
 left_eye_brow_left_y = y[k,15]
 left_eye_brow_right_x = y[k,12]
 left_eye_brow_right_y = y[k,13]
 
 right_eye_brow_left_x = y[k,16]
 right_eye_brow_left_y = y[k,17]
 right_eye_brow_right_x = y[k,18]
 right_eye_brow_right_y = y[k,19]
 
 left_eye_x = y[k,2]
 left_eye_y = y[k,3]
 right_eye_x = y[k,0]
 right_eye_y = y[k,1]

 nose_center_x = y[k,20]
 nose_center_y = y[k,21]

 left_eye_right_corner_x = y[k,4]
 left_eye_right_corner_y = y[k,5]
 left_eye_left_corner_x =  y[k,6]
 left_eye_left_corner_y =  y[k,7]

 right_eye_right_corner_x = y[k,10]
 right_eye_right_corner_y = y[k,11]
 right_eye_left_corner_x =  y[k,8]
 right_eye_left_corner_y =  y[k,9]

 
 nose_center = np.array([nose_center_x,nose_center_y],np.float32)
 
 left_eye = np.array([left_eye_x, left_eye_y], np.float32)
 right_eye = np.array([right_eye_x, right_eye_y], np.float32)
 
 mouth_left = np.array([mouth_left_x, mouth_left_y], np.float32)
 mouth_right = np.array([mouth_right_x, mouth_right_y], np.float32)
 mouth_top  = np.array([mouth_top_x,mouth_top_y],np.float32)
 mouth_bottom = np.array([mouth_bottom_x, mouth_bottom_y],np.float32)

 left_eye_right_corner = np.array([left_eye_right_corner_x,left_eye_right_corner_y])
 left_eye_left_corner = np.array([left_eye_left_corner_x,left_eye_left_corner_y])
 right_eye_right_corner = np.array([right_eye_right_corner_x,right_eye_right_corner_y]) 
 right_eye_left_corner = np.array([right_eye_left_corner_x,right_eye_left_corner_y])

 left_eye_brow_left = np.array([left_eye_brow_left_x,left_eye_brow_left_y],np.float32)
 left_eye_brow_right = np.array([left_eye_brow_right_x,left_eye_brow_right_y],np.float32)
 right_eye_brow_left = np.array([right_eye_brow_left_x,right_eye_brow_left_y],np.float32)
 right_eye_brow_right = np.array([right_eye_brow_right_x,right_eye_brow_right_y],np.float32)


 y_features[k,0] = dist(left_eye,right_eye)
 y_features[k,1] = dist(nose_center,left_eye)
 y_features[k,2] = dist(nose_center,right_eye)
 y_features[k,3] = dist(nose_center,mouth_left)
 y_features[k,4] = dist(nose_center,mouth_right) 
 y_features[k,5] = dist(nose_center, mouth_top)
 y_features[k,6] = dist(left_eye,left_eye_left_corner)
 y_features[k,7] = dist(left_eye, left_eye_right_corner)
 y_features[k,8] = dist(right_eye,right_eye_left_corner)
 y_features[k,9] = dist(right_eye,right_eye_right_corner)
 y_features[k,10] = dist(left_eye_left_corner, left_eye_brow_left)
 y_features[k,11] = dist(left_eye_right_corner, left_eye_brow_right)
 y_features[k,12] = dist(right_eye_left_corner, right_eye_brow_left)
 y_features[k,13] = dist(right_eye_right_corner, right_eye_brow_right)
 y_features[k,14] = dist(mouth_top, mouth_bottom)


print 'Distance Features', y_features.shape
print y_features.reshape(-1,15)

#y = np.append(y, y_features,1)
y = y / 96

print 'Y shape', y.shape

# Extracting Images

df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
X = np.vstack (df['Image'].values) 

X = X.reshape(-1,96,96)

# Histogram equalization
for i in range(len(X)):
       X[i, :, :] = image_histogram_equalization(X[i, :,:])[0]


X = X.astype(np.float32)
X = X/255 
X = X.reshape(-1,1,96,96)

print 'X:', X.shape

'''
# Shrink the image size

I = []

for k in xrange(len(X)):
 img = np.fromstring(X[k], sep=' ', count=96*96)
 I.append(ravel(transform.resize(img.reshape(96,96), (24,24) )))


I = np.asarray(I, 'float32')
X = I/255
X = X.reshape(-1,1,24,24)

#print 'Output X', X.shape
'''


print 'Shape', 'Labels', X.shape, y.shape

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,labels, test_size=0.30)

X_train = X[:1600]
y_train = y[:1600]
X_test = X[1600:]
y_test = y[1600:]

print 'Train, Test shapes (X,y):', X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Train data
f = h5py.File("facialkp-train.hd5", "w")
f.create_dataset("data", data=X_train,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=y_train,  compression="gzip", compression_opts=4)
f.close()

#Test data

f = h5py.File("facialkp-test.hd5", "w")
f.create_dataset("data", data=X_test,  compression="gzip", compression_opts=4)
f.create_dataset("label", data=y_test,  compression="gzip", compression_opts=4)
f.close()


