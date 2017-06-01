# plot the result of super classes and layer and time

import numpy as np
from matplotlib import pylab as plt


load_data = np.load('resultPredictTimeLayer000.npz')
load_dataClass = np.load('resultClassNumber.npz')
load_data_totalAccuracy = np.load('totalAccuracyWithClasses.npz')

resultBinary = load_data['resultBinary']
resultNonZero = load_data['resultNonZero']
layerNumberList = load_data['layerNumberList']
predictTimeList = load_data['predictTimeList']
resultClasses = load_dataClass['resultClasses']
classNumberList = load_dataClass['classNumberList']
totalAccuracy = load_data_totalAccuracy['totalAccuracy']

# plt.figure()
# for i in range(5):
#     plt.plot(predictTimeList, resultBinary[:,i],label='Binary with layer number '+str(i+1))
#     plt.plot(predictTimeList, resultNonZero[:,i],label='Non-zero with layer number '+str(i+1))
# plt.title('Accuracy related to layer numbers and historical time tags')
# plt.xlabel('Historical time tags')
# plt.ylabel('Accuracy')
# plt.legend(loc='right')
# # plt.show()
# plt.grid(True)
# plt.savefig('layerTime.png',dpi=1000)
# plt.close()

# plt.figure()
# for i in range(10):
#     plt.plot(layerNumberList, resultBinary[i,:])
#     plt.plot(layerNumberList, resultNonZero[i,:])
# # plt.title(featureName[i] + ' distribution with zero')
# # plt.xlabel('Value')
# # plt.ylabel('Quantity')
# # plt.savefig('plot/' + featureName[i] + ' value distribution with zero.png', dpi=1000)
# plt.show()
# plt.close()

#
plt.figure()
plt.plot(classNumberList,resultClasses,linewidth=2.0,label='Non-zero accuracy')
plt.plot(classNumberList,totalAccuracy,linewidth=2.0,label='Total accuracy')
plt.legend(loc='right')
plt.xlabel('Super classes number')
plt.ylabel('Accuracy')
plt.title('Accuracy related to super classes number')
plt.grid(True)
# plt.show()
plt.savefig('plot/accuracy.png',dpi=1000)
plt.close()