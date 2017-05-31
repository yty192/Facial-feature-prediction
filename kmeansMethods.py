# A trial of k means

from sklearn.cluster import KMeans
import numpy as np
import theano.tensor as T

# Load data and set train and test
load_data = np.load('arrangedData.npz')

dataAll = load_data['dataAll']
labelData = load_data['labelData']
dataUnify = load_data['dataUnify']

featureNumber = 17
predictTime = 1  # How many previous time do we use to predict next second
classNumber = 100  # How many class in non-zero labels

# up round the labels with the class number of non zero class
roundResolution = 100.0 / float(classNumber)
labelDataRound = np.ceil(labelData / roundResolution)

# one data set contains samples of features and one label
class dataSet:
    def __init__(self):
        self.samples = np.zeros((predictTime,featureNumber-1))
        self.label = 0.0

# length of all data
dataAllLength = dataAll.shape[0]

# length of all data set
dataSetListLength = dataAllLength - predictTime

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
dataSetList = [dataSet() for i in range(dataAllLength - predictTime)]

# fill up the list of all data set
for i in range(len(dataSetList)):
    dataSetList[i].samples = dataUnify[i:i+predictTime,:]
    dataSetList[i].label = labelDataRound[i+predictTime]

# shuffle the list of data set
np.random.shuffle(dataSetList)

# construct the tensors and labels
dataKmeans = np.zeros((dataSetListLength, featureNumber-1))
nonZeroCount = -1
for i in range(len(dataSetList)):
    if predictTime == 1:
        dataKmeans[i,:] = dataSetList[i].samples
    dataSetTensor[i, :, :] = dataSetList[i].samples
    labelList[i] = dataSetList[i].label
    if dataSetList[i].label == 0:
        labelListBinary[i] = dataSetList[i].label
    else:
        nonZeroCount = nonZeroCount + 1
        labelListBinary[i] = 1
        labelListNonZero[nonZeroCount] = dataSetList[i].label
        dataSetTensorNonZero[nonZeroCount,:,:] = dataSetList[i].samples


kmeans = KMeans(n_clusters=101, random_state=2, tol=0.0001).fit(dataKmeans)
# recon_cost = T.nnet.binary_crossentropy(kmeans.labels_, labelListBinary).mean()
same = 0
for i in range(len(labelListBinary)):
    if kmeans.labels_[i] == int(labelList[i]):
        same = same + 1
print  float(same) / float(len(labelListBinary))
# kmeans.predict([[0, 0], [4, 4]])
