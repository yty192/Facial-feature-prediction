# This script test the super classes number

import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import InputLayer, Flatten, LSTM

# Load data and set train and test
load_data = np.load('arrangedData.npz')

dataAll = load_data['dataAll']
labelData = load_data['labelData']
dataUnify = load_data['dataUnify']
fileLength = load_data['fileLength']
totalSamples = sum(fileLength)

featureNumber = 17
predictTime = 2  # How many previous time do we use to predict next second
layerNumber = 3   # How many hidden layers
# classNumber = 20  # How many class in non-zero labels
classNumberList = [2,5,10,20,50,70,100]
resultClasses = np.zeros(len(classNumberList))

classIndex = -1
for classNumber in classNumberList:
    classIndex = classIndex + 1

    # up round the labels with the class number of non zero class
    roundResolution = 100.0 / float(classNumber)
    labelDataRound = np.ceil(labelData / roundResolution)

    # one data set contains samples of features and one label
    class dataSet:
        def __init__(self):
            self.samples = np.zeros((predictTime,featureNumber-1))
            self.label = 0.0


    # length of all data set
    dataSetListLength = sum(fileLength - predictTime)

    # list of all data set
    dataSetList = [dataSet() for i in range(dataSetListLength)]

    # fill up the list of all data set
    fileStart = 0
    countdataset = -1
    for i in fileLength:
        for j in range(fileStart, fileStart + i - predictTime):
            countdataset = countdataset + 1
            dataSetList[countdataset].samples = dataUnify[j:j + predictTime, :]
            dataSetList[countdataset].label = labelDataRound[j + predictTime]
        fileStart = fileStart + i

    # shuffle the list of data set
    np.random.shuffle(dataSetList)

    # take 75 percent as training, 25 percent as testing
    boundaryIndex = int(np.round(0.75 * dataSetListLength))
    dataTrain = dataSetList[:boundaryIndex]
    dataTest = dataSetList[boundaryIndex:]

    # construct the tensors and labels
    # input of single FNN system
    x_all_train = np.zeros((len(dataTrain), predictTime, featureNumber - 1))
    x_all_test = np.zeros((len(dataTest), predictTime, featureNumber - 1))

    # input of binary FNN
    x_binary_train = np.zeros((len(dataTrain), predictTime, featureNumber - 1))
    x_binary_test = np.zeros((len(dataTest), predictTime, featureNumber - 1))

    # label of single FNN system
    y_all_train = np.zeros(len(dataTrain))
    y_all_test = np.zeros(len(dataTest))

    # label of binary FNN
    y_binary_train = np.zeros(len(dataTrain))
    y_binary_test = np.zeros(len(dataTest))

    # get the number of non-zero labels
    # traning
    nonZeroCountTrain = 0
    for i in range(len(dataTrain)):
        if dataTrain[i].label != 0:
            nonZeroCountTrain = nonZeroCountTrain + 1
    # testing
    nonZeroCountTest = 0
    for i in range(len(dataTest)):
        if dataTest[i].label != 0:
            nonZeroCountTest = nonZeroCountTest + 1

    # input and labels of categorical FNN
    x_nonZero_train = np.zeros((nonZeroCountTrain, predictTime, featureNumber - 1))
    x_nonZero_test = np.zeros((nonZeroCountTest, predictTime, featureNumber - 1))
    y_nonZero_train = np.zeros(nonZeroCountTrain)
    y_nonZero_test = np.zeros(nonZeroCountTest)

    # construct all inputs and labels
    # training part
    nonZeroCount = -1
    for i in range(len(dataTrain)):
        x_binary_train[i, :, :] = dataTrain[i].samples
        x_all_train[i, :, :] = dataTrain[i].samples
        y_all_train[i] = dataTrain[i].label
        # labelList[i] = dataSetList[i].label
        if dataTrain[i].label == 0:
            y_binary_train[i] = dataTrain[i].label
        else:
            nonZeroCount = nonZeroCount + 1
            y_binary_train[i] = 1
            y_nonZero_train[nonZeroCount] = dataTrain[i].label
            x_nonZero_train[nonZeroCount, :, :] = dataTrain[i].samples

    # testing part
    nonZeroCount = -1
    for i in range(len(dataTest)):
        x_binary_test[i, :, :] = dataTest[i].samples
        x_all_test[i, :, :] = dataTest[i].samples
        y_all_test[i] = dataTest[i].label
        # labelList[i] = dataSetList[i].label
        if dataTest[i].label == 0:
            y_binary_test[i] = dataTest[i].label
        else:
            nonZeroCount = nonZeroCount + 1
            y_binary_test[i] = 1
            y_nonZero_test[nonZeroCount] = dataTest[i].label
            x_nonZero_test[nonZeroCount, :, :] = dataTest[i].samples

    # non zero FNN
    y_nonZero_train_vectors = keras.utils.to_categorical(y_nonZero_train - 1, num_classes=classNumber)
    y_nonZero_test_vectors = keras.utils.to_categorical(y_nonZero_test - 1, num_classes=classNumber)
    modelNonZero = Sequential()
    modelNonZero.add(InputLayer(input_shape=(x_nonZero_train.shape[1], x_nonZero_train.shape[2])))
    modelNonZero.add(Flatten())
    for i in range(layerNumber):
        modelNonZero.add(Dense(128, activation='relu', name="dense" + str(i + 1)))
    modelNonZero.add(Dense(classNumber, activation='softmax', name="output"))

    modelNonZero.compile(loss='categorical_crossentropy',
                         optimizer='adam',
                         metrics=['accuracy'])

    # modelNonZero.summary()
    earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
    modelNonZero.fit(x_nonZero_train, y_nonZero_train_vectors,
                     epochs=10,
                     batch_size=256,
                     validation_data=(x_nonZero_test, y_nonZero_test_vectors),
                     callbacks=[earlyStop])
    scoreNonZero = modelNonZero.evaluate(x_nonZero_test, y_nonZero_test_vectors, batch_size=256)
    # predictResult = modelNonZero.predict(x_nonZero_test, batch_size=256)
    # resultNonZero[predictIndex, layerIndex] = scoreNonZero[1]
    # modelNonZero.save(filepath="model/nonZeroModel_predictTime" + str(predictTime) + "_layerNumber" + str(layerNumber) + ".hdf5")
    modelNonZero.save(filepath="model/nonZeroModel_classNumber" + str(classNumber) + ".hdf5")
    resultClasses[classIndex] = scoreNonZero[1]
    # print (scoreNonZero)


np.savez('resultClassNumber',
         resultClasses=resultClasses,
         classNumberList=classNumberList)