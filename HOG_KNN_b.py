import numpy as np # linear algebra
from skimage.feature import hog
from skimage.io import ImageCollection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pandas as pd


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

def convert_1D(img_array):
    tmpshape = np.shape(img_array)
    img_size = tmpshape[1] * tmpshape[2]
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

def greyScale(img_array):
    data_gray=[]
    for image in img_array:
        data_gray.append(image[:,:,0])

    print(np.shape(data_gray))
    return data_gray

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

def featureExtraction_HOG(data_gray):
    ppc = 16
    hog_images = []
    hog_features = []
    tmpcount=0
    for image in data_gray:
        fd,hog_image = hog(image, orientations=8, pixels_per_cell=(ppc,ppc),cells_per_block=(4, 4),block_norm= 'L2',visualize=True)
        hog_images.append(hog_image)
        hog_features.append(fd)
        print(tmpcount/30,'%')
        tmpcount=tmpcount+1
    return hog_features

def KNNClassifier(X_train, y_train, X_test,k):

    #Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train) # Fit KNN model


    Y_pred = neigh.predict(X_test)
    return Y_pred


#main code
dataPath= 'dataset'
images,labels=load_dataset(dataPath)
images=greyScale(images)
hog_features=featureExtraction_HOG(images)
labels=convertLabelToBinary(labels)


print('now splitting into training and testing sets, this may also take a minute or two')
xTrain,xTest,yTrain,yTest=train_test_split(hog_features,labels)
print('dimension of xTrain')
print(np.shape(xTrain))
print('dimension of xTest')
print(np.shape(xTest))
print('dimension of yTrain')
print(np.shape(yTrain))
print('dimension of yTest')
print(np.shape(yTest))

#tuning the hyper-parameter K
numberofKs=10
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
