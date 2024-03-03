from experiment_5 import cubeFrames as cf5
from experiment_6 import cubeFrames as cf6
from experiment_7 import cubeFrames as cf7
from experiment_8 import cubeFrames as cf8
from experiment_9 import cubeFrames as cf9
from experiment_10 import cubeFrames as cf10
from experiment_11 import cubeFrames as cf11
from experiment_12 import cubeFrames as cf12

from gaussian import Gaussian, plotGaussian

import asyncio
import sys
import signal
import time
from frame2d import Frame2D
from matplotlib import pyplot as plt
from math import pi, sqrt, pow
import numpy as np
from scipy import *

file_names = [cf5, cf6, cf7, cf8, cf9, cf10, cf11, cf12]
dist = [450, 400, 350, 300, 250, 200, 150, 100]
def printList(dataList, logFile, end="\n"):
    print("[", file=logFile)
    for idx in range(len(dataList)):
        n = num_frames[idx]
        d = dist[idx]
        t = dataList[idx][0]
        v = dataList[idx][1]
        print("("+str(d)+","+str(n)+","+str(t)+", "+str(v)+")", end="", file=logFile)
        if idx != len(dataList)-1:
            print(",", file=logFile)
    print("]", file=logFile, end=end)


def calc_mean_ect(data):
    data2: Frame2D = data[:][:]
    length = len(data2)
    print("length" + str(length))
    mX = [valX[1].x() for valX in data2[:][:]]
    print("x value " + str(mX))
    mY = [valY[1].y() for valY in data2[:][:]]
    print("y value " + str(mY))
    mA = [valA[1].angle() for valA in data2[:][:]]
    print("A value " + str(mA))

    meanX, meanY, meanA = sum(mX) / length, sum(mY) / length, sum(mA) / length
    print("meanX: " + str(meanX) + "meanY: " + str(meanY) + "meanA: " + str(meanA))

    # Calculate Deviations for X,Y,A
    varianceX = sqrt(sum([abs((valX - meanX)) ** 2 for valX in mX]) / length)
    varianceY = sqrt(sum([abs((valY - meanY)) ** 2 for valY in mY]) / length)
    varianceA = sqrt(sum([abs((valA - meanA)) ** 2 for valA in mA]) / length)

    print("varianceX: " + str(varianceX) + "varianceY: " + str(varianceY) + "varianceA: " + str(varianceA))

    return [meanX, meanY, meanA], [varianceX, varianceY, varianceA], length


means = []
variancess = []
num_frames = []
i = 0;
logFile = open("cube_means_variance.py", 'w')
for name in file_names:
    # print(" list: " + str(lists[0][0]))
    print("file number " + str(i))
    M, V, N = calc_mean_ect(name[3][:])
    i += 1
    means.append(M)
    variancess.append(V)
    num_frames.append(N)

print("cubemeans = ", file=logFile, end="")
printList(means, logFile)

print("cubevariance = ", file=logFile, end="")
printList(variancess, logFile)
"""
print("yay? " + str(means) + "double yay!" + str(variancess))

g1 = Gaussian(np.array([means[0][0]]), np.array([[variancess[0][0]]]))
xs = np.linspace(460, 480, 100)
ysg1 = 1 / (sqrt(2 * pi * g1.var[0, 0])) * np.exp(- (xs - g1.mean[0]) ** 2 / (2 * g1.var[0, 0]))
plt.plot(xs, ysg1)
plt.show()

g2 = Gaussian(np.array([means[1][0]]), np.array([[variancess[1][0]]]))
xs2 = np.linspace(410, 430, 100)
ysg2 = 1 / (sqrt(2 * pi * g2.var[0, 0])) * np.exp(- (xs2 - g2.mean[0]) ** 2 / (2 * g2.var[0, 0]))
plt.plot(xs2, ysg2)
plt.show()

g3 = Gaussian(np.array([means[2][0]]), np.array([[variancess[2][0]]]))
xs3 = np.linspace(360, 380, 100)
ysg3 = 1 / (sqrt(2 * pi * g3.var[0, 0])) * np.exp(- (xs3 - g3.mean[0]) ** 2 / (2 * g3.var[0, 0]))
plt.plot(xs3, ysg3)
plt.show()

g4 = Gaussian(np.array([means[3][0]]), np.array([[variancess[3][0]]]))
xs4 = np.linspace(310, 330, 100)
ysg4 = 1 / (sqrt(2 * pi * g4.var[0, 0])) * np.exp(- (xs4 - g4.mean[0]) ** 2 / (2 * g4.var[0, 0]))
plt.plot(xs4, ysg4)
plt.show()

g5 = Gaussian(np.array([means[4][0]]), np.array([[variancess[4][0]]]))
xs5 = np.linspace(270, 290, 100)
ysg5 = 1 / (sqrt(2 * pi * g5.var[0, 0])) * np.exp(- (xs5 - g5.mean[0]) ** 2 / (2 * g5.var[0, 0]))
plt.plot(xs5, ysg5)
plt.show()

g6 = Gaussian(np.array([means[5][0]]), np.array([[variancess[5][0]]]))
xs6 = np.linspace(220, 240, 100)
ysg6 = 1 / (sqrt(2 * pi * g6.var[0, 0])) * np.exp(- (xs6 - g6.mean[0]) ** 2 / (2 * g6.var[0, 0]))
plt.plot(xs6, ysg6)
plt.show()

g7 = Gaussian(np.array([means[6][0]]), np.array([[variancess[6][0]]]))
xs7 = np.linspace(170, 190, 100)
ysg7 = 1 / (sqrt(2 * pi * g7.var[0, 0])) * np.exp(- (xs7 - g7.mean[0]) ** 2 / (2 * g7.var[0, 0]))
plt.plot(xs7, ysg7)
plt.show()

g8 = Gaussian(np.array([means[7][0]]), np.array([[variancess[7][0]]]))
xs8 = np.linspace(120, 140, 100)
ysg8 = 1 / (sqrt(2 * pi * g8.var[0, 0])) * np.exp(- (xs8 - g8.mean[0]) ** 2 / (2 * g8.var[0, 0]))
plt.plot(xs8, ysg8)
plt.show()


# Get first index of a list
def getfirst(ls):
    return [i[0] for i in ls]


meanFirst = getfirst(means)
VarFirst = getfirst(variancess)

plt.plot(meanFirst, VarFirst)
plt.title("Means vs Variances")
plt.xlabel("Mean")
plt.ylabel("Variances")
plt.show()
"""

#-------------BAYES THEOREM---------------------

#def bayes_theorem(p_a, p_b_given_a, p_b_given_not_a):
#	# calculate P(not A)
#	not_a = 1 - p_a
#	# calculate P(B)
#	p_b = p_b_given_a * p_a + p_b_given_not_a * not_a
#	# calculate P(A|B)
#	p_a_given_b = (p_b_given_a * p_a) / p_b
#	return p_a_given_b


# P(A)
#p_a = 0.0001
# P(B|A)
#p_b_given_a = 0.95
# P(B|not A)
#p_b_given_not_a = 0.0001
# calculate P(A|B)
#result = bayes_theorem(p_a, p_b_given_a, p_b_given_not_a)
#print('P(A|B) = %.3f%%' % (result * 100))