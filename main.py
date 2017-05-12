# 0 pitch
# 1 roll
# 2 yaw
# 3 brow_raiser_left
# 4 brow_raiser_right
# 5 brow_lowerer_left
# 6 brow_lowerer_right
# 7 smile
# 8 kiss
# 9 mouth_open
# 10 tongue_out
# 11 eyes_closed_left
# 12 eyes_closed_right
# 13 eyes_turn_left
# 14 eyes_turn_right
# 15 eyes_up
# 16 eyes_down

import numpy as np
from matplotlib import pylab as plt

dataNumber = 99
featureNumber = 17

# list of all feature name
featureName = ['pitch','roll','yaw','brow_raiser_left','brow_raiser_right','brow_lowerer_left','brow_lowerer_right','smile','kiss','mouth_open','tongue_out','eyes_closed_left','eyes_closed_right','eyes_turn_left','eyes_turn_right','eyes_up','eyes_down']

# interesting data for each feature
class featureData:
    Avg = np.zeros(dataNumber)
    Var = np.zeros(dataNumber)
    Max = np.zeros(dataNumber)
    Min = np.zeros(dataNumber)

# feature list of all features with interesting data
featureList = [featureData() for i in range(featureNumber)]

# extract the interesting data and save in feature list
count = -1
for i in range(0,110):
    try:
        data=np.loadtxt('data/f'+str(i).zfill(3)+'.csv', delimiter=',',skiprows=1)
        count = count + 1
    except:
        print(i)
        continue
    for j in range(featureNumber):
        featureList[j].Avg[count] = np.mean(data[:,j])
        featureList[j].Var[count] = np.var(data[:,j])
        featureList[j].Max[count] = np.max(data[:,j])
        featureList[j].Min[count] = np.min(data[:,j])
        plt.figure()
        plt.boxplot(data[:,j])
        plt.title('boxplot '+featureName[j]+' of file ' + str(i))
        plt.savefig('plot/boxplot '+featureName[j]+' of file ' + str(i) + '.png')
        plt.close()




# plotting
# plt.ion()
# plt.interactive(False)
for i in range(featureNumber):
    plt.figure()
    plt.plot(featureList[i].Avg)
    plt.title('average ' + featureName[i])
    plt.savefig('plot/average ' + featureName[i] + '.png')
    plt.close()
    plt.figure()
    plt.plot(featureList[i].Var)
    plt.title('variance ' + featureName[i])
    plt.savefig('plot/variance ' + featureName[i] + '.png')
    plt.close()
# plt.show()



