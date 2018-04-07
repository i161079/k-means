import re
import codecs
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(inFile):
    inDate = codecs.open(inFile, 'r', 'utf-8').readlines()
    dataSet = list()
    for line in inDate:
        line = line.strip()
        strList = re.split('[ ]+', line)
        numList = list()
        for item in strList:
            num = float(item)
            numList.append(num)
        dataSet.append(numList)
    return dataSet

def initCentroids(dataSet, k):
    numSamples, dim = dataSet.shape
    centroids = np.zeros((k, dim))
    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i, :] = dataSet[index, :]
    return centroids

def kmeans(in_dataSet, k, e):
    numSamples, dim = in_dataSet.shape
    dataSet = np.zeros((numSamples, dim+1))
    dataSet[:, 0:dim] = in_dataSet
    centroids = initCentroids(dataSet, k)
    for _ in range(e):
        for i in range(numSamples):
            minDist = 100000.0
            minIndex = 0
            for j in range(k):
                distance = np.linalg.norm(centroids[j, :] - dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            dataSet[i, -1] = minIndex
        for j in range(k):
            pointsInCluster = dataSet[dataSet[:, -1] == j, :-1]
            centroids[j, :-1] = np.mean(pointsInCluster, axis=0)
    return dataSet, centroids

testX = np.array(loadDataSet('test.txt'))
dataSet,_ = kmeans(testX, 4, 100)
print(dataSet)
