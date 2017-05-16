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

import numpy as np
import collections
from matplotlib import pylab as plt

dataNumber = 99
featureNumber = 17

# list of all feature name
featureName = ['pitch','roll','yaw','brow_raiser_left','brow_raiser_right','brow_lowerer_left','brow_lowerer_right','smile','kiss','mouth_open','tongue_out','eyes_closed_left','eyes_closed_right','eyes_turn_left','eyes_turn_right','eyes_up','eyes_down']

# interesting data for each feature
class featureData:
    def __init__(self):
        self.Avg = np.zeros(dataNumber)
        self.Var = np.zeros(dataNumber)
        self.Max = np.zeros(dataNumber)
        self.Min = np.zeros(dataNumber)


# feature list of all features with interesting data
featureList = [featureData() for i in range(featureNumber)]

# data for box plot
boxPlotData = np.zeros((1,featureNumber))

# data of all files
dataAll = []

# list of all smile data
smileData = np.zeros(1)
# extract the interesting data and save in feature list
count = -1
for i in range(111):
    try:
        data=np.loadtxt('data/f'+str(i).zfill(3)+'.csv', delimiter=',',skiprows=1)
        count = count + 1
    except:
        print(i)
        continue
    for j in range(featureNumber):
        # featureList[j].Avg[count] = np.mean(data[:,j+1])
        # featureList[j].Var[count] = np.var(data[:,j+1])
        # featureList[j].Max[count] = np.max(data[:,j+1])
        # featureList[j].Min[count] = np.min(data[:,j+1])
        # if featureName[j] == 'smile':
        #     smileData = np.concatenate([smileData,data[:,j+1]])
        # plt.figure()
        # plt.boxplot(data[:,j+1])
        # plt.title('boxplot '+featureName[j]+' of file ' + str(i))
        # plt.savefig('plot/boxplot '+featureName[j]+' of file ' + str(i) + '.png')
        # plt.close()
        boxPlotTemp = data[:,j+1]
        boxPlotTemp.shape = (-1,1)
        # if boxPlotData[j] == 0:
        #     boxPlotData[j] = boxPlotTemp
        boxPlotData[j] = [boxPlotData[j],boxPlotTemp]
    if i == 0:
        dataAll = data[:,1:]
    else:
        dataAll = np.vstack((dataAll,data[:,1:]))

# smileCounter = collections.Counter(smileData)
# print(smileCounter.keys()[1:])
# print(smileCounter.values()[1:])

# plotting
# plt.ion()
# plt.interactive(False)

# plt.figure()
# plt.stem(smileCounter.keys()[1:],smileCounter.values()[1:])
# plt.title('smile value distribution')
# plt.savefig('plot/smile value distribution.png')
# plt.show()
# plt.close()

# for i in range(featureNumber):
#     plt.figure()
#     plt.plot(featureList[i].Avg)
#     plt.title('average ' + featureName[i])
#     plt.savefig('plot/average ' + featureName[i] + '.png')
#     plt.close()
#     plt.figure()
#     plt.plot(featureList[i].Var)
#     plt.title('variance ' + featureName[i])
#     plt.savefig('plot/variance ' + featureName[i] + '.png')
#     plt.close()




