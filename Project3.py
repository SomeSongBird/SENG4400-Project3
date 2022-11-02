import math
import pandas as pd

# Building block functions:
    
def dist(x1, x2):
    """ Return euclidean distance between x1 and x2 """
    dist_sqr = 0
    for i in range(len(x1)):
        dist_sqr += (x1[i]-x2[i])**2
    dist = math.sqrt(dist_sqr)
    return dist

def centroid(xList):
    """ Compute centroid of multi-dimensional x data in xList """
    centroid = []*len(xList[0]) # set the centroid to have the same dimentions as a point
    for i in range(len(centroid)):
        centroid[i] = sum([x[i] for x in xList])/len(xList)
    return centroid

def assignmentDiffers(yCurrent, yPrev):
    """ Return True if yCurrent differs from yPrev """
    # TODO
    pass

# Let's use a class for our K-Means implementation
class KMeans:
    """ Perform k-means clustering """
    
    def __init__(self, k=5):
        self.k = k          # number of clusters
        self.means = None   # means of clusters
        
    def classify(self, x):
        """ Return the index of the cluster closest to the input """
        # TODO
        pass
    
    def train(self, data):
        """ Train model based on data """
        # TODO assign to self.means, one per cluster
        pass


def kmeans(x, k):
    # Use it like this?
    km = KMeans(k = 10)
    km.train(x)
    # can print out km.means to see the fit means
    # can call km.classify([1,2,3,4]) to get cluster index
    
    #The function should return a list the length of x that contains
    # the cluster number (1 - k) for the corresponding x point
    # TODO determine return value

def test():
    li = [[1,2],[3,4],[5,6]]
    print(sum([x[0] for x in li]))

