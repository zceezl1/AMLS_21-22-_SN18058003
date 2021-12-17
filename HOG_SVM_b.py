import numpy as np # linear algebra
from skimage.feature import hog
from skimage.io import ImageCollection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import time
import pandas as pd

#import dataset
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


#main code
dataPath= 'dataset'
images,labels=load_dataset(dataPath)
images=greyScale(images)
hog_features=featureExtraction_HOG(images)
labels=convertLabelToBinary(labels)

#training model
clf=OneVsRestClassifier(LinearSVC(C=100))
#clf = svm.SVC()
hog_features = np.array(hog_features)

X_train,X_test,Y_train,Y_test=train_test_split(hog_features,labels)

print('start training')
starttime=time.time()
clf.fit(X_train,Y_train)
print('training done')
endtime=time.time()
print('time taken: ',endtime-starttime)

print('predicting with testing set')
Y_pred = clf.predict(X_test)
print("Accuracy: "+str(accuracy_score(Y_test, Y_pred)))
print('\n')
print(classification_report(Y_test, Y_pred))