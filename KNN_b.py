import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import ImageCollection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def load_dataset(dataPath):
    #define the path of the data and the label
    imagePath= dataPath+'/image'
    labelPath= dataPath+'/label.csv'

    #load labels
    labels=pd.read_csv(labelPath)
    #using only the useful part of the csv file
    labels=labels['label']

    #load image files
    print('loading images, this may take up to a minute')
    img_array=ImageCollection(imagePath + '/IMAGE_*.jpg')
    print('images loaded\n')
    return img_array,labels

def convertLabelToBinary(labels):
    tmpsize = np.shape(labels)
    # convert the labels to binary (tumor or no tumor)
    tmplabel = np.zeros(tmpsize[0])
    for label in range(0, tmpsize[0]):
        if labels[label] == 'no_tumor':
            tmplabel[label] = False
        else:
            tmplabel[label] = True
    return(tmplabel)

def convert_1D(img_array):
    tmpshape = np.shape(img_array)
    img_size = tmpshape[1] * tmpshape[2] * tmpshape[3]
    img_size_grey = tmpshape[1] * tmpshape[2]
    feature_array = np.zeros((tmpshape[0], img_size))
    #turn to grey scale 1D array
    for image in range(0,tmpshape[0]):
        tmpimage=img_array[image]
        tmpimage=tmpimage[:,:,0]

        img_1D_vector = tmpimage.reshape(img_size_grey)

        start=0#image*img_size
        end=img_size_grey#(image+1)*img_size

        feature_array[image,start:end]=img_1D_vector

        percentage=(image+1)/tmpshape[0]
        percentage = percentage * 100
        if percentage%10==0:
            print(percentage,'%')
    return feature_array

#define classifier
def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred


#main code
#split data into training and testing sets
dataPath= 'dataset'
images,labels=load_dataset(dataPath)
images=convert_1D(images)
labels=convertLabelToBinary(labels)

print('now splitting into training and testing sets, this may also take a minute or two')
xTrain,xTest,yTrain,yTest=train_test_split(images,labels)
print('dimension of xTrain')
print(np.shape(xTrain))
print('dimension of xTest')
print(np.shape(xTest))
print('dimension of yTrain')
print(np.shape(yTrain))
print('dimension of yTest')
print(np.shape(yTest))

#tuning the hyper-parameter K
numberofKs=5
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
    print(classification_report(yTest, tmpY))

import matplotlib.pyplot as plt
print(K)
print(tmpScore)
print(np.shape(K))
print(np.shape(tmpScore))
plt.plot(K,tmpScore)
plt.show()
