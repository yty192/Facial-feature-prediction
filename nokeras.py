# This script build FNN, testing with different layer and time steps

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
# predictTime = 1
predictTimeList = range(1,11)  # How many previous time do we use to predict next second
layerNumberList = range(1,6)   # How many hidden layers
classNumber = 10  # How many class in non-zero labels
classNumberList = [2,5,10,20,50,70,100]
totalAccuracy = np.zeros(len(classNumberList))
iterationNumber = 1

classIndex = -1
for iteration in range(iterationNumber):
# for classNumber in classNumberList:
    classIndex = classIndex + 1

    resultBinary = np.zeros((len(predictTimeList),len(layerNumberList)))
    resultNonZero = np.zeros((len(predictTimeList),len(layerNumberList)))

    # up round the labels with the class number of non zero class
    roundResolution = 100.0 / float(classNumber)
    labelDataRound = np.ceil(labelData / roundResolution)

    predictIndex = -1
    for predictTime in predictTimeList:
        predictIndex = predictIndex + 1
        # one data set contains samples of features and one label
        class dataSet:
            def __init__(self):
                self.samples = np.zeros((predictTime,featureNumber-1))
                self.label = 0.0

        # length of all data
        dataAllLength = dataAll.shape[0]

        # length of all data set
        dataSetListLength = sum(fileLength - predictTime)

        # length of non zero label data set
        nonZeroLength = np.count_nonzero(labelDataRound)

        # tensor of data sets corresponding to all label (for binary training)
        dataSetTensor = np.zeros((dataSetListLength, predictTime, featureNumber-1))

        # tensor of data set corresponding to non zero labels
        dataSetTensorNonZero = np.zeros((nonZeroLength,predictTime,featureNumber-1))

        # list of labels corresponding to tensor of data sets
        labelList = np.zeros(dataSetListLength)

        # list of binary labels
        labelListBinary = np.zeros(dataSetListLength)

        # list of non zero labels
        labelListNonZero = np.zeros(nonZeroLength)

        # list of all data set
        dataSetList = [dataSet() for i in range(dataSetListLength)]

        # fill up the list of all data set
        fileStart = 0
        countdataset = -1
        for i in fileLength:
            for j in range(fileStart,fileStart+i-predictTime):
                countdataset = countdataset + 1
                dataSetList[countdataset].samples = dataUnify[j:j + predictTime, :]
                dataSetList[countdataset].label = labelDataRound[j + predictTime]
            fileStart = fileStart + i
        # for i in range(len(dataSetList)):
        #     dataSetList[i].samples = dataUnify[i:i+predictTime,:]
        #     dataSetList[i].label = labelDataRound[i+predictTime]

        # shuffle the list of data set
        np.random.shuffle(dataSetList)

        # take 75 percent as training, 25 percent as testing
        boundaryIndex = int(np.round(0.75 * dataSetListLength))
        dataTrain = dataSetList[:boundaryIndex]
        dataTest = dataSetList[boundaryIndex:]

        # construct the tensors and labels
        x_all_train = np.zeros((len(dataTrain), predictTime, featureNumber-1))
        x_all_test = np.zeros((len(dataTest), predictTime, featureNumber - 1))
        x_binary_train = np.zeros((len(dataTrain), predictTime, featureNumber - 1))
        x_binary_test = np.zeros((len(dataTest), predictTime, featureNumber - 1))
        y_all_train = np.zeros(len(dataTrain))
        y_all_test = np.zeros(len(dataTest))
        y_binary_train = np.zeros(len(dataTrain))
        y_binary_test = np.zeros(len(dataTest))

        nonZeroCountTrain = 0
        for i in range(len(dataTrain)):
            if dataTrain[i].label != 0:
                nonZeroCountTrain = nonZeroCountTrain + 1

        nonZeroCountTest = 0
        for i in range(len(dataTest)):
            if dataTest[i].label != 0:
                nonZeroCountTest = nonZeroCountTest + 1

        x_nonZero_train = np.zeros((nonZeroCountTrain, predictTime, featureNumber - 1))
        x_nonZero_test = np.zeros((nonZeroCountTest, predictTime, featureNumber - 1))
        y_nonZero_train = np.zeros(nonZeroCountTrain)
        y_nonZero_test = np.zeros(nonZeroCountTest)

        nonZeroCount = -1
        for i in range(len(dataTrain)):
            x_binary_train[i,:,:] = dataTrain[i].samples
            x_all_train[i,:,:] = dataTrain[i].samples
            y_all_train[i] = dataTrain[i].label
            # labelList[i] = dataSetList[i].label
            if dataTrain[i].label == 0:
                y_binary_train[i] = dataTrain[i].label
            else:
                nonZeroCount = nonZeroCount + 1
                y_binary_train[i] = 1
                y_nonZero_train[nonZeroCount] = dataTrain[i].label
                x_nonZero_train[nonZeroCount,:,:] = dataTrain[i].samples

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

        layerIndex = -1
        for layerNumber in layerNumberList:
            layerIndex = layerIndex + 1

            # binary classification
            # build a forward neural network
            model = Sequential()
            model.add(InputLayer(input_shape=(x_binary_train.shape[1], x_binary_train.shape[2])))
            model.add(Flatten())
            for i in range(layerNumber):
                model.add(Dense(128, activation='relu', name="dense"+str(i+1)))
            model.add(Dense(1, activation='sigmoid', name="output"))

            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # model.summary()
            earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
            model.fit(x_binary_train, y_binary_train,
                      epochs=10,
                      batch_size=256,
                      validation_data=(x_binary_test, y_binary_test),
                      callbacks=[earlyStop])
            scoreBinary = model.evaluate(x_binary_test, y_binary_test, batch_size=256)
            resultBinary[predictIndex,layerIndex] = scoreBinary[1]
            # model.save(filepath="model/binaryModel_predictTime"+str(predictTime)+"_layerNumber"+str(layerNumber)+".hdf5")

            # print (scoreBinary)


            # non zero classification
            y_nonZero_train_vectors = keras.utils.to_categorical(y_nonZero_train - 1, num_classes=classNumber)
            y_nonZero_test_vectors = keras.utils.to_categorical(y_nonZero_test - 1, num_classes=classNumber)
            # build a forward neural network
            modelNonZero = Sequential()
            modelNonZero.add(InputLayer(input_shape=(x_nonZero_train.shape[1], x_nonZero_train.shape[2])))
            modelNonZero.add(Flatten())
            for i in range(layerNumber):
                modelNonZero.add(Dense(128, activation='relu', name="dense"+str(i+1)))
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
                             callbacks= [earlyStop])
            scoreNonZero = modelNonZero.evaluate(x_nonZero_test, y_nonZero_test_vectors, batch_size=256)
            # predictResult = modelNonZero.predict(x_nonZero_test, batch_size=256)
            resultNonZero[predictIndex, layerIndex] = scoreNonZero[1]
            # modelNonZero.save(filepath="model/nonZeroModel_predictTime" + str(predictTime) + "_layerNumber" + str(layerNumber) + ".hdf5")

            # print (scoreNonZero)


            # # all data classification
            #
            # y_all_train_vectors = keras.utils.to_categorical(y_all_train, num_classes=101)
            # y_all_test_vectors = keras.utils.to_categorical(y_all_test, num_classes=101)
            # # build a forward neural network
            # modelAll = Sequential()
            # modelAll.add(InputLayer(input_shape=(x_all_train.shape[1], x_all_train.shape[2])))
            # modelAll.add(Flatten())
            # for i in range(layerNumber):
            #     modelAll.add(Dense(128, activation='relu', name="dense" + str(i + 1)))
            # modelAll.add(Dense(101, activation='softmax', name="output"))
            #
            # modelAll.compile(loss='categorical_crossentropy',
            #                      optimizer='adam',
            #                      metrics=['accuracy'])
            #
            # # modelAll.summary()
            # earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
            #                                           mode='auto')
            # modelAll.fit(x_all_train, y_all_train_vectors,
            #                  epochs=10,
            #                  batch_size=256,
            #                  validation_data=(x_all_test, y_all_test_vectors),
            #                  callbacks=[earlyStop])
            # scoreAll = modelAll.evaluate(x_all_test, y_all_test_vectors, batch_size=256)
            # predictResult = modelAll.predict(x_All_test, batch_size=256)
            # resultAll[predictIndex, layerIndex] = scoreAll[1]
            # modelAll.save(filepath="model/allModel_predictTime" + str(predictTime) + "_layerNumber" + str(layerNumber) + ".hdf5")

            # print (scoreAll)

            # # Stacked LSTM for sequence classification
            #
            #
            # # input and label
            # x_nonZero = dataSetTensorNonZero
            # y_nonZero = labelListNonZero
            #
            # # 75 percent are training data, 25 percent are testing data
            # boundaryIndexNonZero = np.round(0.75*x_nonZero.shape[0])
            # x_nonZero_train = x_nonZero[:boundaryIndexNonZero,:,:]
            # y_nonZero_train = keras.utils.to_categorical(y_nonZero[:boundaryIndexNonZero]-1, num_classes=classNumber)
            # x_nonZero_test = x_nonZero[boundaryIndexNonZero:,:,:]
            # y_nonZero_test = keras.utils.to_categorical(y_nonZero[boundaryIndexNonZero:]-1, num_classes=classNumber)
            #
            #
            # # expected input data shape: (batch_size, timesteps, data_dim)
            # modelLSTM = Sequential()
            # modelLSTM.add(LSTM(32, return_sequences=True,
            #                input_shape=(predictTime, featureNumber-1)))  # returns a sequence of vectors of dimension 32
            # modelLSTM.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
            # modelLSTM.add(LSTM(32))  # return a single vector of dimension 32
            # modelLSTM.add(Dense(classNumber, activation='softmax'))
            #
            # modelLSTM.compile(loss='categorical_crossentropy',
            #               optimizer='rmsprop',
            #               metrics=['accuracy'])
            # modelLSTM.summary()
            # modelLSTM.fit(x_nonZero_train, y_nonZero_train,
            #           batch_size=128, epochs=5,
            #           validation_data=(x_nonZero_test, y_nonZero_test))


            # for i in range(x_all_test.shape[0]):
            #     model.predict(x_all_test[i])

    # prediction result of binary model
    binaryPredict = model.predict(x_binary_test)
    binaryPredict[binaryPredict>0.5] = 1
    binaryPredict[binaryPredict<0.5] = 0
    binaryPredict = binaryPredict.flatten()

    # how many 0 and 1 are correctly predicted
    countCorrectBinaryWithOne = 0
    countCorrectBinaryWithZero = 0
    for i in range(len(binaryPredict)):
        if binaryPredict[i] == y_binary_test[i] and y_binary_test[i] == 1:
            countCorrectBinaryWithOne = countCorrectBinaryWithOne + 1
        if binaryPredict[i] == y_binary_test[i] and y_binary_test[i] == 0:
            countCorrectBinaryWithZero = countCorrectBinaryWithZero + 1

    # construct the data set to do the categorical prediction from the correctly predicted data by binary model
    x_nonZeroTobePredict = np.zeros((countCorrectBinaryWithOne, predictTime, featureNumber-1))
    y_nonZeroTobePredict = np.zeros(countCorrectBinaryWithOne)
    countCorrectBinaryWithOne = -1
    for i in range(len(binaryPredict)):
        if binaryPredict[i] == y_binary_test[i] and y_binary_test[i] == 1:
            countCorrectBinaryWithOne = countCorrectBinaryWithOne + 1
            x_nonZeroTobePredict[countCorrectBinaryWithOne,:,:] = x_nonZero_test[countCorrectBinaryWithOne,:,:]
            y_nonZeroTobePredict[countCorrectBinaryWithOne] = y_nonZero_test[countCorrectBinaryWithOne]

    y_nonZeroTobePredict_vectors = keras.utils.to_categorical(y_nonZeroTobePredict-1, num_classes=classNumber)
    scoreNonZeroAfterBinary = modelNonZero.evaluate(x_nonZeroTobePredict, y_nonZeroTobePredict_vectors, batch_size=256)

    countCorrectNonZero = scoreNonZeroAfterBinary[1] * (countCorrectBinaryWithOne + 1)

    totalAccuracy[classIndex] = float(countCorrectNonZero + countCorrectBinaryWithZero) / float(len(y_binary_test))

# print totalAccuracy
# np.savez('totalAccuracyWithClasses',totalAccuracy=totalAccuracy)
    np.savez('result/resultPredictTimeLayer000',
            resultBinary=resultBinary,
            resultNonZero=resultNonZero,
            layerNumberList=layerNumberList,
            predictTimeList=predictTimeList)