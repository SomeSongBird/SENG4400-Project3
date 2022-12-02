import math
import pandas as pd
import random
import time

# Building block functions:
    
def deepcopy(clist):
    returnlist = []
    for partlist in clist:
        returnlist.append(partlist.copy())
    return returnlist

def dist(x1, x2):
    """ Return euclidean distance between x1 and x2 """
    dist_sqr = 0
    for i in range(len(x1)):
        dist_sqr += (x1[i]-x2[i])**2
    dist = math.sqrt(dist_sqr)
    return dist

def centroid(xList):
    """ Compute centroid of multi-dimensional x data in xList """
    center = xList.sum()/len(xList)
    return center.tolist()

def assignmentDiffers(yCurrent, yPrev):
    """ Return True if yCurrent differs from yPrev """

    for i in range(len(yCurrent)):
        # make sure they're the same length
        if len(yCurrent[i])!=len(yPrev[i]):
            #print("length difference: "+str(i))
            return True
        # check if each item is equal
        for j in range(len(yCurrent[i])):
            if yCurrent[i][j]!=yPrev[i][j]:
                #print("value difference\ncurrent"+str(yCurrent[i][j])+"\nprevious\n"+str(yPrev[i][j]))
                return True
    return False

# Let's use a class for our K-Means implementation
class KMeans:
    """ Perform k-means clustering """
    
    def __init__(self, k=5):
        self.k = k          # number of clusters
        self.means = []  # means of clusters
        self.MAX_ITTERATIONS = 50
        
    def classify(self, x):
        """ Return the index of the cluster closest to the input """
        smallest_dist = None
        smallest_index = 0
        for i in range(len(self.means)):
            current_dist = dist(x,self.means[i])
            if smallest_dist==None:
                smallest_dist = current_dist
                continue
            if smallest_dist > current_dist:
                smallest_dist = current_dist
                smallest_index = i
        return smallest_index
    
    def train(self, data):
        """ Train model based on data """
        # Initial random cluster assignment
        initial_clusters = random.sample(range(len(data)), self.k)
        
        for x in range(self.k):
            self.means.append(data.iloc[initial_clusters[x]].tolist())
        
        # place the data's index into a 2d array.  each 1st dimention index is a cluster
        clusters = [[] for ind in range(self.k)]
        new_clusters = [[] for ind in range(self.k)]
        assignChange=True
        itterations = 0
        while assignChange and itterations<=self.MAX_ITTERATIONS:
            itterations+=1
            #print(itterations)
            
            # preform training
            for index in range(len(data)):
                classification = self.classify(data.iloc[index])
                new_clusters[classification].append(index)
                
            # check if any changes occured in the clustering
            assignChange=assignmentDiffers(new_clusters,clusters)
            # update the cluster centroids to reflect the updated clusters
            if assignChange:
                for i in range(self.k):
                    self.means[i] = centroid(data.iloc[new_clusters[i]])
                clusters = deepcopy(new_clusters)
                new_clusters = [[] for ind in range(self.k)]
        

class DBSCAN:
    """ Perform DBSCAN clustering """
    def __init__(self,epsilon,minPts):
        self.epsilon = epsilon
        self.minPts = minPts
        self.model = None
    
    def classify(self,x):
        """ Return the index of the cluster to which this point would belong, or None """
        clusters = {}
        justData = self.model.loc[:,self.model.columns!='cluster']
        #print(justData)
        #return None
        for point in self.model.index:
            if(dist(x,justData.iloc[point])<=self.epsilon):
                clust = self.model.iloc[point]['cluster']
                if(clust!=-1):
                    try:
                        clusters[clust]+=1
                    except e:
                        clusters[clust] = 1
        if clusters!={}:
            return max(clusters,key=clusters.get) # returns the key with the highest value
        return -1

    def findFirstUnassigned(self, data):
        for dat in data.index:
            if data["cluster"][dat]==-1:
                return dat  # return the index of the first instance of a point without a cluster assignment
        return None

    def train(self,data):
        """ Train model based on data """

        corePoints = [] # the index of all core points
        nonCorePoints = [] # the index of all non-core points

        assignmentData = data.copy()
        assignmentData["cluster"] = -1 # setting the default cluster of every point to an "unassigned" value

        for pt in data.index:
            #print(pt)
            # count points in radius from pt
            closePts = 0
            for pt2 in data.index[pt:]:
                #print(pt2)
                if pt == pt2:
                    continue
                dis = dist(data.iloc[pt],data.iloc[pt2])
                if dis <= self.epsilon:
                    closePts += 1
            if closePts >= self.minPts:
                corePoints.append(pt)
            elif closePts >0:
                nonCorePoints.append(pt)
        
        #print(corePoints)

        # assigning the core points to clusters 
        # the first cluster isn't random because I don't see a value in it being random.
        # to change it to use a random, change the function below to get a list of every unassigned core point then return a random value from that
        cluster=0
        pt = self.findFirstUnassigned(assignmentData.iloc[corePoints])
        while pt!=None:

            # List of every point added to the cluster to make sure every core point around them gets added as well
            currentCluster = [pt]  # list init
            assignmentData["cluster"][pt] = cluster  
            currentClusterIndex=0  # current place in the cluster

            # loop until every point that has been added has been used
            while currentClusterIndex<len(currentCluster):
                for pt2Index in corePoints:
                    if dist(data.iloc[pt],data.iloc[pt2Index])<=self.epsilon:
                        # checks if no cluster has been assigned to the point
                        if assignmentData["cluster"][pt2Index] == -1:
                            currentCluster.append(pt2Index)  #adds it to the list
                            assignmentData["cluster"][pt2Index] = cluster
                # itterates to the next point in the cluster
                currentClusterIndex += 1
                # error check
                if currentClusterIndex<len(currentCluster):
                    pt = currentCluster[currentClusterIndex]
                else:
                    break
            # updates to the next cluster name and resets the starting point
            #print(currentCluster)
            cluster+=1
            pt = self.findFirstUnassigned(assignmentData.iloc[corePoints])
        
        # assigning the edge points to clusters
        for pt in nonCorePoints:
            # in case of multiple (but not more than min) core points being close to the edge point
            # in the case of a tie it's up to whatever the max function makes it
            clusters = {}
            for corePtIndex in corePoints:
                if dist(data.iloc[pt],data.iloc[corePtIndex]) <= self.epsilon:
                    try:
                        clusters[assignmentData["cluster"][corePtIndex]]+=1
                    except:
                        clusters[assignmentData["cluster"][corePtIndex]] = 1
            # sets the cluster to the max value 
            # !!potential for bugs here!!
            if(clusters!={}):
                assignmentData["cluster"][pt] = max(clusters,key=clusters.get)

        self.model = assignmentData.copy()

def kmeans(x, k):
    # Use it like this?
    km = KMeans(k)
    km.train(x)
    print(km.means)

def dbScan(data,epsilon,minPts):
    dbscan = DBSCAN(epsilon,minPts)
    dbscan.train(data.iloc[:500])
    #print(dbscan.model['cluster'].unique())
    classif = dbscan.classify([1,1])
    print(classif)  

def main():
    path=""
    infile="clustering_dataset_01.csv"
    df = pd.read_csv(path+infile,header=None)
    #kmeans(df[df.columns[:-1]],5)
    dbScan(df[df.columns[:-1]][:500],1,3)

if __name__=="__main__":
    main()