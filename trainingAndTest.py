# This script build FNN, testing with different layer and time steps

import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import InputLayer, Flatten, LSTM

# Load data
load_data = np.load('arrangedData.npz')

dataAll = load_data['dataAll']
labelData = load_data['labelData']
dataUnify = load_data['dataUnify']
fileLength = load_data['fileLength']
totalSamples = sum(fileLength)

featureNumber = 17
# predictTime = 1
predictTimeList = range(3,4)  # How many previous time do we use to predict next second
layerNumberList = range(4,5)   # How many hidden layers
classNumber = 100  # How many class in non-zero labels
classNumberList = [2,5,10,20,50,70,100]
totalAccuracy = np.zeros(len(classNumberList))
iterationNumber = 1

classIndex = -1
for iteration in range(iterationNumber):
# for classNumber in classNumberList:
    classIndex = classIndex + 1

    # save results of different historical time tags and layer numbers
    resultBinary = np.zeros((len(predictTimeList),len(layerNumberList)))
    resultNonZero = np.zeros((len(predictTimeList),len(layerNumberList)))

    # up round the labels with the class number of non zero class
    roundResolution = 100.0 / float(classNumber)
    labelDataRound = np.ceil(labelData / roundResolution)

    predictIndex = -1
    for predictTime in predictTimeList:
        predictIndex = predictIndex + 1

        # one data set contains one sample of features and one label
        class dataSet:
            def __init__(self):
                self.samples = np.zeros((predictTime,featureNumber-1))
                self.label = 0.0

        # length of all data set
        dataSetListLength = sum(fileLength - predictTime)

        # list of all data set
        dataSetList = [dataSet() for i in range(dataSetListLength)]

        # fill up the list of all data set through each file with number historical time tags
        fileStart = 0
        countDataset = -1
        for i in fileLength:
            for j in range(fileStart,fileStart+i-predictTime):
                countDataset = countDataset + 1
                dataSetList[countDataset].samples = dataUnify[j:j + predictTime, :]
                dataSetList[countDataset].label = labelDataRound[j + predictTime]
            fileStart = fileStart + i

        # shuffle the list of data set
        np.random.shuffle(dataSetList)

        # take 75 percent as training (dataTrain), 25 percent as testing (dataTest)
        boundaryIndex = int(np.round(0.75 * dataSetListLength))
        dataTrain = dataSetList[:boundaryIndex]
        dataTest = dataSetList[boundaryIndex:]

        # construct the tensors and labels
        # input of single FNN system
        x_all_train = np.zeros((len(dataTrain), predictTime, featureNumber-1))
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

        # Build FNN
        layerIndex = -1
        for layerNumber in layerNumberList:
            layerIndex = layerIndex + 1

            # binary FNN
            modelBinary = Sequential()
            modelBinary.add(InputLayer(input_shape=(x_binary_train.shape[1], x_binary_train.shape[2])))
            modelBinary.add(Flatten())
            for i in range(layerNumber):
                modelBinary.add(Dense(128, activation='relu', name="dense"+str(i+1)))
            modelBinary.add(Dense(1, activation='sigmoid', name="output"))

            modelBinary.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])

            # model.summary()
            earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
            modelBinary.fit(x_binary_train, y_binary_train,
                      epochs=10,
                      batch_size=256,
                      validation_data=(x_binary_test, y_binary_test),
                      callbacks=[earlyStop])
            scoreBinary = modelBinary.evaluate(x_binary_test, y_binary_test, batch_size=256)
            resultBinary[predictIndex,layerIndex] = scoreBinary[1]
            # model.save(filepath="model/binaryModel_predictTime"+str(predictTime)+"_layerNumber"+str(layerNumber)+".hdf5")

            # print (scoreBinary)


            # non zero FNN
            y_nonZero_train_vectors = keras.utils.to_categorical(y_nonZero_train - 1, num_classes=classNumber)
            y_nonZero_test_vectors = keras.utils.to_categorical(y_nonZero_test - 1, num_classes=classNumber)
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


            # single FNN for all classes
            y_all_train_vectors = keras.utils.to_categorical(y_all_train, num_classes=101)
            y_all_test_vectors = keras.utils.to_categorical(y_all_test, num_classes=101)
            # build a forward neural network
            modelAll = Sequential()
            modelAll.add(InputLayer(input_shape=(x_all_train.shape[1], x_all_train.shape[2])))
            modelAll.add(Flatten())
            for i in range(layerNumber):
                modelAll.add(Dense(128, activation='relu', name="dense" + str(i + 1)))
            modelAll.add(Dense(101, activation='softmax', name="output"))

            modelAll.compile(loss='categorical_crossentropy',
                                 optimizer='adam',
                                 metrics=['accuracy'])

            # modelAll.summary()
            earlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0,
                                                      mode='auto')
            modelAll.fit(x_all_train, y_all_train_vectors,
                             epochs=10,
                             batch_size=256,
                             validation_data=(x_all_test, y_all_test_vectors),
                             callbacks=[earlyStop])
            scoreAll = modelAll.evaluate(x_all_test, y_all_test_vectors, batch_size=256)
            # predictResult = modelAll.predict(x_All_test, batch_size=256)
            # resultAll[predictIndex, layerIndex] = scoreAll[1]
            # modelAll.save(filepath="model/allModel_predictTime" + str(predictTime) + "_layerNumber" + str(layerNumber) + ".hdf5")

            # print (scoreAll)



            # # A trial of Stacked LSTM for sequence classification
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



    # prediction result of binary model
    binaryPredict = modelBinary.predict(x_binary_test)
    binaryPredict[binaryPredict>0.5] = 1
    binaryPredict[binaryPredict<0.5] = 0
    binaryPredict = binaryPredict.flatten()

    # prediction result of single FNN model
    allPredict = modelAll.predict(x_all_test)
    allPredict = np.argmax(allPredict, axis=1)

    # how many 0 and 1 are correctly predicted
    countCorrectBinaryWithOne = 0
    countCorrectBinaryWithZero = 0

    # how many 1 are wrongly predicted as 0
    countWrongZeroTwoFNN = 0
    countWrongZeroOneFNN = 0

    for i in range(len(binaryPredict)):
        if binaryPredict[i] == y_binary_test[i] and y_binary_test[i] == 1:
            countCorrectBinaryWithOne = countCorrectBinaryWithOne + 1
        if binaryPredict[i] == y_binary_test[i] and y_binary_test[i] == 0:
            countCorrectBinaryWithZero = countCorrectBinaryWithZero + 1
        if binaryPredict[i] == 0 and y_binary_test[i] == 1:
            countWrongZeroTwoFNN = countWrongZeroTwoFNN + 1
        if allPredict[i] == 0 and y_all_test[i] != 0:
            countWrongZeroOneFNN = countWrongZeroOneFNN + 1

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
#     np.savez('result/resultPredictTimeLayer000',
#             resultBinary=resultBinary,
#             resultNonZero=resultNonZero,
#             layerNumberList=layerNumberList,
#             predictTimeList=predictTimeList)