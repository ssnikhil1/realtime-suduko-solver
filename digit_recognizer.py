import numpy as np
import cv2
import os
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator



##creating dataset for training
path = 'myData/myData'
count = 0
images = []
classNo = []
myList=os.listdir(path)

noOfClasses = len(myList)

for x in range(0, noOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg, (32, 32))
        images.append(curImg)
        classNo.append(x)



images = np.array(images)
classNo = np.array(classNo)

##splitting the data into train,validation and test dataset
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=0.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)

##checking distribution of each class using bar chart
numOfSamples = []
for x in range(0, noOfClasses):
    numOfSamples.append(len(np.where(y_train == x)[0]))

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No of Images for each Class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
plt.show()


##image preprocessing function for training
def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img



##preprocessing train,test,validation data for training and testing purpose
X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

##image augmentation to reduce chances of overfitting
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

##one hot encoding of target variables
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)



def myModel():

    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(60, sizeOfFilter1, input_shape=(32,32, 1), activation='relu')))
    model.add((Conv2D(60, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(30, sizeOfFilter2, activation='relu')))
    model.add((Conv2D(30, sizeOfFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model=myModel()
history = model.fit_generator(dataGen.flow(X_train,y_train, batch_size=50),steps_per_epoch=2,epochs=1,validation_data=(X_validation,y_validation),shuffle=1)
##plotting results
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()


score = model.evaluate(X_test, y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])
"""
pickle_out= open("model_trained_10.p", "wb")
pickle.dump(model,pickle_out)
pickle_out.close()"""

##testing model
width = 640
height = 480
threshold = 0.65
cameraNo = 0

cap = cv2.VideoCapture(cameraNo)
cap.set(3, width)
cap.set(4, height)

pickle_in = open("model_trained_10.p", "rb")
model = pickle.load(pickle_in)

while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img, (32, 32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image", img)
    img = img.reshape(1, 32, 32, 1)
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    if probVal > threshold:
        cv2.putText(imgOriginal, str(classIndex) + "   " + str(probVal),(50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    cv2.imshow("Original Image", imgOriginal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break






