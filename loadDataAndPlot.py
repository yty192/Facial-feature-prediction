# This script load all data, plot the statistics, save unified data for training and testing

# 0 time
# 1 pitch
# 2 roll
# 3 yaw
# 4 brow_raiser_left
# 5 brow_raiser_right
# 6 brow_lowerer_left
# 7 brow_lowerer_right
# 8 smile
# 9 kiss
# 10 mouth_open
# 11 tongue_out
# 12 eyes_closed_left
# 13 eyes_closed_right
# 14 eyes_turn_left
# 15 eyes_turn_right
# 16 eyes_up
# 17 eyes_down


import time
import numpy as np
import collections
from matplotlib import pylab as plt

start_time = time.time()

fileNumber = 99
featureNumber = 17
isPlot = False

# list of all feature name
featureName = ['pitch','roll','yaw','brow_raiser_left','brow_raiser_right','brow_lowerer_left','brow_lowerer_right','smile','kiss','mouth_open','tongue_out','eyes_closed_left','eyes_closed_right','eyes_turn_left','eyes_turn_right','eyes_up','eyes_down']

# interesting statistics of each feature
class featureStatistics:
    def __init__(self):
        self.Avg = np.zeros(fileNumber)
        self.Var = np.zeros(fileNumber)
        self.Std = np.zeros(fileNumber)
        self.Max = np.zeros(fileNumber)
        self.Min = np.zeros(fileNumber)


# feature list of all features with interesting statistics
featureList = [featureStatistics() for i in range(featureNumber)]

# data for box plot
boxPlotData = [[] for i in range(featureNumber)]

# data of all files
dataAll = []

# list of all smile data
smileData = np.zeros(0)

# list of time data
timeData = np.zeros(0)

# list of file length
fileLength = []

# extract the interesting data and save in feature list
count = -1
for i in range(111):
    # print(i)
    try:
        data=np.loadtxt('data/f'+str(i).zfill(3)+'.csv', delimiter=',',skiprows=1)
        fileLength.append(int(data.shape[0]))
        count = count + 1
    except:
        #print(i)
        continue
    for j in range(featureNumber):
        featureList[j].Avg[count] = np.mean(data[:,j+1])
        featureList[j].Var[count] = np.var(data[:,j+1])
        featureList[j].Std[count] = np.std(data[:,j+1])
        featureList[j].Max[count] = np.max(data[:,j+1])
        featureList[j].Min[count] = np.min(data[:,j+1])
        if featureName[j] == 'smile':
            smileData = np.concatenate([smileData,data[:,j+1]])

        if isPlot:
            # boxplot of each feature of each file
            plt.figure()
            plt.boxplot(data[:,j+1])
            plt.title('boxplot '+featureName[j]+' of file ' + str(i))
            plt.savefig('plot/boxplot/boxplot '+featureName[j]+' of file ' + str(i) + '.png', dpi=1000)
            plt.close()

        boxPlotData[j].append(data[:,j+1])

    timeData = np.concatenate([timeData,data[:,0]])
    if i == 0:
        dataAll = data[:,1:]
    else:
        dataAll = np.vstack((dataAll,data[:,1:]))

if isPlot:
    # plot statistics of each feature
    for i in range(featureNumber):
        print(i)
        # box plot of each feature of all files
        plt.figure(figsize=(35, 10))
        plt.boxplot(boxPlotData[i])
        plt.title('boxplot of ' + featureName[i])
        plt.xlabel('File number')
        plt.savefig('plot/box plot of ' + featureName[i] + '.png',dpi=1000)
        plt.close()

        # average value of each feature
        plt.figure()
        plt.plot(featureList[i].Avg)
        plt.title('average ' + featureName[i])
        plt.xlabel('File number')
        plt.savefig('plot/average ' + featureName[i] + '.png',dpi=1000)
        plt.close()

        # variance of each feature
        plt.figure()
        plt.plot(featureList[i].Var)
        plt.title('variance ' + featureName[i])
        plt.xlabel('File number')
        plt.savefig('plot/variance ' + featureName[i] + '.png',dpi=1000)
        plt.close()

        # plot distribution of some features
        if i > 2:
            distributionData = dataAll[:,i]
            valueCounter = collections.Counter(distributionData)
            plt.figure()
            plt.stem(valueCounter.keys(),valueCounter.values())
            plt.title(featureName[i]+' distribution with zero')
            plt.xlabel('Value')
            plt.ylabel('Quantity')
            plt.savefig('plot/'+featureName[i]+' value distribution with zero.png',dpi=1000)
            plt.close()
            plt.figure()
            plt.stem(valueCounter.keys()[1:],valueCounter.values()[1:])
            plt.xlabel('Value')
            plt.ylabel('Quantity')
            plt.title(featureName[i]+' distribution without zero')
            plt.savefig('plot/'+featureName[i]+' distribution without zero.png',dpi=1000)
            plt.close()


# standardization of data
dataUnify = dataAll
labelData = smileData
dataUnify = np.delete(dataUnify,featureName.index('smile'),1)
for i in range(featureNumber-1):
    dataUnify[:,i] = (dataAll[:,i] - np.mean(dataAll[:,i])) / np.std(dataAll[:,i])

# eigenvalues of the covariance matrix of all features
w, v = np.linalg.eig(np.dot(dataUnify.T, dataUnify))

# sort the eigenvalues
w_sort = np.sort(w)

np.savez('arrangedData',
         dataAll=dataAll,
         timeData=timeData,
         labelData=labelData,
         dataUnify=dataUnify,
         w=w,
         w_sort=w_sort,
         featureNumber=featureNumber,
         fileLength=fileLength)

end_time = time.time()
elapsed_time = end_time - start_time
print elapsed_time