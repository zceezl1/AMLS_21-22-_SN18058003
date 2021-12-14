import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import ImageCollection
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
i=1
for file in os.walk('dataset/image'):
    print(i)
    print(numpy.shape(file))
    print('\n')
    i=i+1
print(i)
'''
'''
tmpimage=cv2.imread('dataset/image/IMAGE_0000.jpg')
tmpimage1=tmpimage[:,:,0]-tmpimage[:,:,1]#/3#+tmpimage[:,:,1]/3+tmpimage[:,:,2]/3
cv2.imshow('w',tmpimage1)
cv2.waitKey(1000)
print(sum(tmpimage1))
'''

#define the path of the data and the label
dataPath= 'dataset/image'
labelPath='dataset/label.csv'

#load labels
labels=pd.read_csv(labelPath)
#using only the useful part of the csv file
labels=labels['label']

#load image files
print('loading images, this may take up to a minute')
img_array=ImageCollection(dataPath + '/IMAGE_*.jpg')

tmpshape=np.shape(img_array)
print('images loaded\n')

#maybe extract features before splitting?
#feature extraction
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print('extracting features')
img_size=tmpshape[1]*tmpshape[2]*tmpshape[3]
img_size_grey=tmpshape[1]*tmpshape[2]
feature_array=np.zeros((tmpshape[0],img_size))
for image in range(0,tmpshape[0]):
    tmpimage=img_array[image]
    tmpimage=tmpimage[:,:,0]

    img_1D_vector = tmpimage.reshape(img_size_grey)

    start=0#image*img_size
    end=img_size_grey#(image+1)*img_size

    feature_array[image,start:end]=img_1D_vector

    percentage=(image+1)/tmpshape[0]
    if percentage%0.1==0:
        print(percentage*100,'%')
features=feature_array
#split data into training and testing sets
print('now splitting into training and testing sets, this may also take a minute or two')
xTrain,xTest,yTrain,yTest=train_test_split(features,labels)

print('dimension of xTrain')
print(np.shape(xTrain))
print('dimension of xTest')
print(np.shape(xTest))
print('dimension of yTrain')
print(np.shape(yTrain))
print('dimension of yTest')
print(np.shape(yTest))


'''
for file in range(0,tmpshape[0]):
    tmpimage=img_array[file]

    image_array1[file]=1
'''


#test code
#define classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred

#classification
'''
Y_pred=KNNClassifier(xTrain, yTrain, xTest,4)

score=metrics.accuracy_score(yTest,Y_pred)
print(score)
'''

#tuning the hyper-parameter K
numberofKs=20
print('tuning hyper-parameter K')
K=np.arange(1,numberofKs)
tmpScore=np.empty((numberofKs-1,1))
print(np.shape(tmpScore))
for valueK in K:
    print('value of K:',valueK)
    tmpY=KNNClassifier(xTrain,yTrain,xTest,valueK)
    score=metrics.accuracy_score(yTest,tmpY)
    print('score:',score)
    tmpScore[valueK-1]=score

import matplotlib.pyplot as plt
plt.plot(K,tmpScore)
plt.show()


#test model with tuned hyper-parameter and test set
