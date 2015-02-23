import numpy as np
import matplotlib.pyplot as pl
import pandas as pd
import h5py
from scipy.spatial import distance
from skimage import transform
from numpy import ravel

# Make sure that caffe is on the python path:
#caffe_root = './caffe/'  # this file is expected to be in {caffe_root}/examples
#import sys
#sys.path.append(caffe_root + 'python')

import caffe

# Set the right path to your model definition file, pretrained model weights,
# and the image you would l1ike to classify.
MODEL_FILE = './facialkp_predict.prototxt'
PRETRAINED = './tmp_iter_1000.caffemodel'

def dist(x,y):   
    #return numpy.sqrt(numpy.sum((x-y)**2))
    return distance.cdist(x, y, 'euclidean')

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf

df = pd.read_csv('test.csv',header=0)

df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
X = np.vstack (df['Image'].values) 
X = X.reshape([-1,96,96])
X = X.astype(np.float32)

# Run Histogram equalization
for i in range(len(X)):
       X[i,  :, :] = image_histogram_equalization(X[i,:,:])[0]

'''
I = []

for k in xrange(len(X)):
 img = np.fromstring(X[k], sep=' ', count=96*96)
 I.append(ravel(transform.resize(img.reshape(96,96), (24,24) )))

I = np.asarray(I, 'float32')

'''
# Scale between 0 and 1
X = X/255
X = X.reshape(-1,1,96,96)


BATCH_SIZE = 64
total = len(X)
max_value = 64   #batch_size from .prototxt
batches = abs(len(X) / BATCH_SIZE ) + 1 

data = np.zeros((batches*BATCH_SIZE,1,96,96),np.float32)

data[0:len(X),:] = X

print 'Input Data shape: ', data.shape
print 'Total batches: ', batches


net = caffe.Net(MODEL_FILE,PRETRAINED)
net.set_mode_gpu()


data4D = np.zeros([max_value,1,96,96],np.float32)
data4DL = np.zeros([max_value,1,1,1], np.float32)

data4D = X[400:464,:,:,:]

net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
pred = net.forward()
ip1 = net.blobs['ip2'].data * 96

print 'Predicted', ip1
print 'Shape', ip1.shape


predicted = []

for b in xrange(batches): 

 data4D = np.zeros([BATCH_SIZE,1,96,96]) #create 4D array, first value is batch_size, last number of inputs
 data4DL = np.zeros([BATCH_SIZE,1,1,1]) # need to create 4D array as output, first value is  batch_size, last number of outputs
 data4D[0:BATCH_SIZE,:] = data[b*BATCH_SIZE:b*BATCH_SIZE+BATCH_SIZE,:] # fill value of input xtrain

 #predict
 #print [(k, v[0].data.shape) for k, v in net.params.items()]
 net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
 pred = net.forward()
 print 'batch ', b, data4D.shape, data4DL.shape

 predicted.append(pred['ip2']*96)


predicted = np.asarray(predicted, 'float32')
predicted = predicted.reshape(batches*BATCH_SIZE,30)
predicted = predicted[:total,:30]

print 'Total in Batches ', data4D.shape, batches
print 'Predicted shape: ', predicted.shape
print 'Saving to csv..'



np.savetxt("fkp_output.csv", predicted, delimiter=",")


for k in xrange(400,412,1):

 y = predicted[k]
 print y.reshape(-1,30)
 
 pl.imshow(df['Image'][k].reshape(96,96), cmap='gray' )
 pl.scatter(y[0::2], y[1::2],  marker='x', s=30)
 pl.show()

